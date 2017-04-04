#!/usr/bin/python3
import os
import random
import itertools
import logging

from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session, relationship
from sqlalchemy import create_engine

import tensorflow as tf
import numpy as np


DATABASE_URI = os.environ["DATABASE_URI"]
MODEL_ROOT = os.path.join(os.getcwd(), os.path.dirname(__file__))\
        + "/models/"

# ORM definitions
Match = Roster = Participant = ParticipantExt = Player = None


def connect():
    global Match, Roster, Participant, ParticipantExt, Player
    # generate schema from db
    Base = automap_base()
    engine = create_engine(DATABASE_URI)
    Base.prepare(engine, reflect=True)

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
    ParticipantExt = Base.classes.participant_ext
    Participant.participant_ext = relationship(
        "participant_ext", foreign_keys="participant_ext.participant_api_id",
        primaryjoin="and_(participant_ext.participant_api_id == participant.api_id)")
    Player = Base.classes.player

    return Session(engine)


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
        if table == "participant_ext":
            return vars(record.participant_ext[0])[column]
        if table == "player":
            return vars(record.player[0])[column]
        if table == "roster":
            return vars(record.roster[0])[column]
        if table == "match":
            return vars(record.roster[0].match[0])[column]
        raise KeyError("Invalid path " + path)

    def _batch(self, ids=[], size=None):
        if len(ids) == 0:  # training, get random sample
            size = size or self.batchsize
            # train a model one batch
            offset = random.random() * self._db.query(Participant).count()
            # have a bit of randomness in the sample
            records = self._db.query(
                Participant).offset(offset).limit(size).all()
        else:
            records = self._db.query(
                Participant).filter(Participant.api_id.in_(ids)).all()

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
        self.features = ["participant.kills", "participant.deaths",
                         "participant.assists"]
        self.label = "participant_ext.rating"
        self.type = "linear"
        self.batches = 1
        self.batchsize = 500
        self.steps = 500
        self.id = "kda-win"
        super().__init__(db)

    def estimate(self, record):
        # for training, rating = participant.winner
        return record.winner


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    db = connect()
    model = MVPScoreModel(db)
    model.train(force=True)

    print(model._model.evaluate(input_fn=model._batch(), steps=1))
