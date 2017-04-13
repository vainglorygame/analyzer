#!/usr/bin/python3
import os
import time
import random
import logging
import itertools

from sqlalchemy.orm import Session, relationship
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine

import tensorflow as tf
import numpy as np

import pika


RABBITMQ_URI = os.environ.get("RABBITMQ_URI") or "amqp://localhost"
DATABASE_URI = os.environ["DATABASE_URI"]
BATCHSIZE = os.environ.get("BATCHSIZE") or 1000  # objects
IDLE_TIMEOUT = os.environ.get("IDLE_TIMEOUT") or 1  # s
MODEL_ROOT = os.path.join(os.getcwd(), os.path.dirname(__file__))\
        + "/models/"

# ORM definitions
Match = Roster = Participant = ParticipantStats = Player = None
db = rabbit = channel = None

# batch storage
queue = []
timer = None

# models
mvpmodel = None


def connect():
    global Match, Roster, Participant, ParticipantStats, Player
    global db, rabbit, channel

    # generate schema from db
    Base = automap_base()
    engine = create_engine(DATABASE_URI)
    while True:
        try:
            Base.prepare(engine, reflect=True)
            break
        except OperationalError as err:
            logging.error(err)
            time.sleep(5)

    # definitions
    # TODO check whether the primaryjoin clause is the best method to do this
    Match = Base.classes.match
    Roster = Base.classes.roster
    Roster.match = relationship(
        "match", foreign_keys="match.api_id",
        primaryjoin="and_(match.api_id == roster.match_api_id)")
    Participant = Base.classes.participant
    Participant.roster = relationship(
        "roster", foreign_keys="roster.api_id",
        primaryjoin="and_(roster.api_id == participant.roster_api_id)")
    Participant.player = relationship(
        "player", foreign_keys="player.api_id",
        primaryjoin="and_(player.api_id == participant.player_api_id)")
    ParticipantStats = Base.classes.participant_stats
    Participant.participant_stats = relationship(
        "participant_stats", foreign_keys="participant_stats.participant_api_id",
        primaryjoin="and_(participant_stats.participant_api_id == participant.api_id)")
    Player = Base.classes.player

    db = Session(engine)

    while True:
        try:
            rabbit = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URI))
            break
        except pika.exceptions.ConnectionClosed as err:
            logging.error(err)
            time.sleep(5)
    channel = rabbit.channel()
    channel.queue_declare(queue="analyze", durable=True)
    channel.basic_qos(prefetch_count=BATCHSIZE)
    channel.basic_consume(newjob, queue="analyze")


class Model(object):
    # override this configuration
    features = []
    label = ""
    type = ""
    batches = 1
    batchsize = 1
    steps = 0
    id = "unlabeled"

    def __init__(self, db):
        self._db = db
        for feat in self.features:
            self._feature_cols = [tf.contrib.layers.real_valued_column(
                feat, dimension=1)]
        if self.type == "linear":
            self._model = tf.contrib.learn.LinearClassifier(
                feature_columns=self._feature_cols,
                model_dir=MODEL_ROOT + self.id,
                config=tf.contrib.learn.RunConfig(
                    save_checkpoints_secs=1))

    def _from_record(self, path, record):
        table, column = path.split(".")
        if table == "participant":
            return vars(record)[column]
        if table == "participant_stats":
            return vars(record.participant_stats[0])[column]
        if table == "player":
            return vars(record.player[0])[column]
        if table == "roster":
            return vars(record.roster[0])[column]
        if table == "match":
            return vars(record.roster[0].match[0])[column]
        raise KeyError("Invalid path " + path)

    def _batch(self, ids=None, size=None):
        if ids is None:  # training, get random sample
            size = size or self.batchsize
            # train a model one batch
            offset = random.random() * self._db.query(Participant).count()
            # have a bit of randomness in the sample
            records = self._db.query(
                Participant).offset(offset).limit(size).all()
        else:
            records = self._db.query(
                Participant).filter(
                    Participant.api_id.in_(ids)).order_by(
                        Participant.api_id.desc()).all()

        data = {}
        labels = []
        # populate from db records
        for record in records:
            labels.append(self.estimate(record))
            for path in self.features:
                if path not in data:
                    data[path] = []
                data[path].append(self._from_record(path, record))

        # convert to numpy arrs
        for key in data:
            data[key] = np.array(data[key])
        labels = np.array(labels)

        logging.info("sample size: %s", len(labels))
        if ids is not None:
            assert len(ids) == len(labels), "got nonexisting participant"

        return tf.contrib.learn.io.numpy_input_fn(
            data, labels, batch_size=self.batchsize,
            num_epochs=self.steps)

    # TODO at the moment, it's tied to Participant
    def train(self, force=False):
        if force or not os.path.isdir(MODEL_ROOT + self.id):
            monitor = tf.contrib.learn.monitors.ValidationMonitor(
                input_fn=self._batch(),
                eval_steps=1, every_n_steps=20)
            for _ in range(self.batches):
                self._model.fit(input_fn=self._batch(),
                                steps=self.steps,
                                monitors=[monitor])

    def predict(self, ids):
        return itertools.islice(
            self._model.predict_proba(input_fn=self._batch(ids)),
            len(ids))

    def estimate(self, record):
        # override: calculate or return the label's value
        pass


class MVPScoreModel(Model):
    def __init__(self, db):
        self.features = ["participant_stats.kills",
                         "participant_stats.deaths",
                         "participant_stats.assists"]
        self.label = "participant_stats.impact_score"
        self.type = "linear"
        self.batches = 1
        self.batchsize = 500
        self.steps = 500
        self.id = "kda-win"
        super().__init__(db)

    def estimate(self, record):
        # for training, rating = participant.winner
        return record.winner

    def rate(self, ids):
        return [w for l, w in self.predict(ids)]


def newjob(_, method, properties, body):
    global timer, queue
    queue.append((method, properties, body))
    if timer is None:
        timer = rabbit.add_timeout(IDLE_TIMEOUT, process)
    if len(queue) == BATCHSIZE:
        process()


def process():
    global timer, queue, db, mvpmodel
    if timer is not None:
        rabbit.remove_timeout(timer)
        timer = None
    jobs = queue[:]
    queue = []

    logging.info("analyzing batch %s", str(len(jobs)))
    ids = list(set([str(id, "utf-8") for _, _, id in jobs]))
    ratings = mvpmodel.rate(ids)

    with db.no_autoflush:
        for c in range(len(ratings)):
            rating = int(100 * float(ratings[c]))
            pstat = db.query(ParticipantStats).filter(
                ParticipantStats.participant_api_id == ids[c]).first()
            # manual upsert
            if pstat is None:
                db.add(ParticipantStats(participant_api_id=ids[c],
                                        impact_score=rating))
            else:
                pstat.impact_score = rating

    db.commit()
    # ack all until this one
    logging.info("acking batch")
    channel.basic_ack(jobs[-1][0].delivery_tag, multiple=True)
    # notify web
    for api_id in ids:
        channel.basic_publish("amq.topic", "participant." + api_id,
                              "stats_update")


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    connect()
    mvpmodel = MVPScoreModel(db)
    mvpmodel.train()
    logging.info(mvpmodel._model.evaluate(input_fn=mvpmodel._batch(), steps=1))

    channel.start_consuming()
