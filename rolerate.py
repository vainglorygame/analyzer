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


DATABASE_URI = os.environ["DATABASE_URI"]
MODEL_ROOT = os.path.join(os.getcwd(), os.path.dirname(__file__))\
        + "/models/"

# ORM definitions
Match = Roster = Participant = ParticipantStats = Player = None
db = None

# models
mvpmodel = None


def connect():
    global db
    global Match, Roster, Participant, Hero, ParticipantStats, Player

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
    Participant.hero = relationship(
        "hero", foreign_keys="hero.id",
        primaryjoin="and_(hero.id == participant.hero_id)")
    Hero = Base.classes.hero
    ParticipantStats = Base.classes.participant_stats
    Participant.participant_stats = relationship(
        "participant_stats", foreign_keys="participant_stats.participant_api_id",
        primaryjoin="and_(participant_stats.participant_api_id == participant.api_id)")
    Player = Base.classes.player

    db = Session(engine)


class RoleModel(object):
    def __init__(self, db):
        self.batches = 15
        self.batchsize = 300
        self.steps = 500
        self.id = "role-kda"

        self._db = db
        self._feature_cols = [
            tf.contrib.layers.real_valued_column("kills_per_min", dimension=1),
            tf.contrib.layers.real_valued_column("deaths_per_min", dimension=1),
            tf.contrib.layers.real_valued_column("assists_per_min", dimension=1)
        ]
        self._model = tf.contrib.learn.LinearClassifier(
            feature_columns=self._feature_cols,
            model_dir=MODEL_ROOT + self.id,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=1))

    def _batch(self, ids=None, size=None):
        if ids is None:  # training, get random sample
            size = size or self.batchsize
            # train a model one batch
            offset = int(random.random() * self._db.query(Participant)\
                .filter(Participant.hero.any(is_captain=True))\
                .count())
            # have a bit of randomness in the sample
            records = self._db.query(Participant)\
                .filter(Participant.hero.any(is_captain=True))\
                .offset(offset).limit(size)\
                .all()
        else:
            records = self._db.query(Participant)\
                .filter(Hero.is_carry)\
                .filter(Participant.api_id.in_(ids))\
                .order_by(Participant.api_id.desc())\
                .all()

        data = {}
        labels = []
        # populate from db records
        data["kills_per_min"] = []
        data["deaths_per_min"] = []
        data["assists_per_min"] = []
        for record in records:
            labels.append(record.winner)
            data["kills_per_min"].append(record.participant_stats[0].kills
                / record.roster[0].match[0].duration)
            data["deaths_per_min"].append(record.participant_stats[0].deaths
                / record.roster[0].match[0].duration)
            data["assists_per_min"].append(record.participant_stats[0].assists
                / record.roster[0].match[0].duration)

        logging.info("---------- %s data points for training ---------", len(labels))
        # convert to numpy arrs
        for key in data:
            data[key] = np.array(data[key])
        labels = np.array(labels)

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


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    connect()
    rolemodel = RoleModel(db)
    rolemodel.train()
    for name in rolemodel._model.get_variable_names():
        logging.info("%s: %s", name,
                     rolemodel._model.get_variable_value(name))
