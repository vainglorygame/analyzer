#!/usr/bin/python

import os
import json
import itertools
import logging
import tensorflow as tf
import psycopg2
import psycopg2.extras


class Classifier(object):
    def __init__(self, name):
        self._name = name
        self.model = None
        self.modeldir = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))) + "/models/" + self._name

        self._conn = None

        # empty defaults
        self._categories = []
        self._continuous = []
        self._classes = []

        # learn configuration
        self._steps = 0  # per batch
        self._batchsize = 0
        self._max_batches = 0  # number of batches to train
        self._dimension = 0  # embeddings dimension, log_2(num of unique features)
        self._bucketsize = 0  # sparse column hash bucket size
        self._hidden = []  # hidden layers http://stats.stackexchange.com/a/1097

        # db SQL mappings
        # TODO use cache
        # override these
        self._trainquery = ""
        self._testquery = ""
        self._insertquery = ""
        self._label = ""

        #
        self._binary = False
        self._label_available = True  # label is in data

    def connect(self, **args):
        self._conn = psycopg2.connect(**args)

    def _model_setup(self):
        """Sets up a model that takes the `num_features` features as input."""
        feature_columns = [
            tf.contrib.layers.sparse_column_with_hash_bucket(feat, self._bucketsize)
            for feat in self._categories
        ]
        emb_columns = [
            tf.contrib.layers.embedding_column(
                sparse_id_column=col,
                dimension=self._dimension
            )
            for col in feature_columns
        ] + [
            tf.contrib.layers.real_valued_column(feat)
            for feat in self._continuous
        ]

        self.model = tf.contrib.learn.DNNClassifier(
            feature_columns=emb_columns,
            hidden_units=self._hidden,
            n_classes=len(self._classes),
            model_dir=self.modeldir,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=2
            )
        )

    def _get_sample(self, size=None, filter=None):
        """Return a data set from the database.
        :param size: (optional) The number of items to get. Defaults to `All`.
        :type size: int or str
        :param offset: (optional) The SQL OFFSET parameter.
        :type offset: int or str
        """
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        assert (filter is None) ^ (size is None)
        if filter is None:
            cur.execute(self._trainquery, (size,))
        else:
            cur.execute(self._testquery, (filter,))

        ids = []
        data = {name: [] for name in self._continuous + [self._label]}
        for rec in cur:
            for key in data.keys():
                if key == self._label and not self._label_available:
                    val = self.guess_label(rec)
                else:
                    val = rec[key]
                data[key].append(val)
                ids.append(rec["id"])
        cur.close()
        return data, ids

    def guess_label(self, data):
        """Return a label based on data."""
        # override
        pass

    # TODO tf warning - dimensions are wrong
    def _toinput(self, sample, train=True):
        """Convert sample to Tensors.
        :param sample: Data dictionary.
        :type sample: dict
        :param train: (optional) Whether to return labels too.
        :type train: bool
        :return: features, label
        :rtype: tuple
        """
        continuous = {k: tf.constant(sample[k])
                      for k in self._continuous}
        categories = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(len(sample[k]))],
            values=sample[k],
            shape=[len(sample[k]), 1])
                    for k in self._categories}

        features = {**continuous, **categories}

        if train:
            label = tf.constant(sample[self._label])
            return features, label
        else:
            return features

    def _more(self):
        """Fetches up to `limit` number of training items
           from the database in batches."""
        # TODO maybe you can use an iterator?
        sample = self._get_sample(size=self._batchsize)[0]
        return self._toinput(sample, train=True)

    def train(self):
        """Train a DNN on a static, algorithmic guess."""
        self._model_setup()

        if os.path.isdir(self.modeldir):
            logging.warning("already trained, not training again")
            return

        # get one batch of testing data
        # TODO either terminate with `eval_steps` or with OutOfRangeError
        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            input_fn=lambda: self._toinput(self._get_sample(size=1000)[0], train=True),
            eval_steps=1,
            every_n_steps=10
        )

        for _ in range(self._max_batches):
            self.model.fit(
                input_fn=lambda: self._more(),
                steps=self._steps,
                monitors=[validation_monitor]
            )

    def classify(self, sample):
        """Classify a data set.

        :param only_best: (optional) Return the predicted result
                          instead of a dict of propabilities.
        :type only_best: bool
        :return: Prediction results.
        :rtype: list of dict or list
        """
        return self.model.predict_proba(input_fn=lambda: self._toinput(sample, train=False))

    def windup(self):
        pass

    def teardown(self, failed=False):
        if failed:
            pass
        else:
            self._conn.commit()

    def classify_db(self, objids):
        """Classify all data in the data base and insert."""
        # split sample (with participant ids) into data and ids
        # TODO use parameters
        sample, ids = self._get_sample(filter=objids)
        d = self.classify(sample)
        cnt = 0
        cur = self._conn.cursor()
        for l in itertools.islice(d, len(objids)):
            # TODO use executemany?
            cur.execute(self._insertquery, {
                "id": ids[cnt],
                self._label: json.dumps(l.tolist())
            })
            cnt += 1
        cur.close()

class KDAClassifier(Classifier):
    """A DNN that classifies loss/win based on KDA."""
    def __init__(self):
        super().__init__("kda-win")

        # DNN configuration
        self._continuous = ["kills", "deaths", "assists", "cs", "teamkills"]
        self._classes = ["loss", "win"]
        self._label = "win"
        self._binary = True

        self._steps = 200
        self._batchsize = 500
        self._max_batches = 5

        self._trainquery = """
            SELECT
                participant.kills,
                participant.deaths,
                participant.assists,
                participant.farm AS cs,
                roster.hero_kills AS teamkills,
                participant.winner AS win,
                participant.api_id AS id
            FROM participant
            TABLESAMPLE BERNOULLI(5)
            JOIN roster on participant.roster_api_id=roster.api_id
            JOIN match on roster.match_api_id=match.api_id
            LIMIT %s
        """
        self._testquery = """
            SELECT
                participant.kills,
                participant.deaths,
                participant.assists,
                participant.farm AS cs,
                roster.hero_kills AS teamkills,
                participant.winner AS win,
                participant.api_id AS id
            FROM participant
            JOIN roster on participant.roster_api_id=roster.api_id
            JOIN match on roster.match_api_id=match.api_id
            WHERE participant.api_id=ANY(%s)
        """
        self._insertquery = """
            INSERT INTO participant_stats
            (patch_version, participant_api_id, score)
            VALUES(2.2, %(id)s, (%(win)s::json->>1)::float)
            ON CONFLICT(participant_api_id) DO
            UPDATE SET score=(%(win)s::json->>1)::float
        """

        self._dimension = 5
        self._bucketsize = 100
        self._hidden = [2]


class RoleClassifier(Classifier):
    """A DNN that classifies participants into roles based on CS."""
    def __init__(self):
        super().__init__("roles")

        # DNN configuration
        self._continuous = ["lanefarm", "junglefarm"]
        self._classes = ["carry", "jungler", "captain"]
        self._label = "role"
        self._binary = True
        self._label_available = False

        self._steps = 200
        self._batchsize = 500
        self._max_batches = 5

        self._trainquery = """
            SELECT
                participant.hero AS hero,
                participant.jungle_kills AS junglefarm,
                (participant.minion_kills-participant.jungle_kills) AS lanefarm,
                participant.api_id AS id
            FROM participant
            TABLESAMPLE BERNOULLI(5)
            LIMIT %s
        """
        self._testquery = """
            SELECT
                participant.hero AS hero,
                participant.jungle_kills AS junglefarm,
                (participant.minion_kills-participant.jungle_kills) AS lanefarm,
                participant.api_id AS id
            FROM participant
            WHERE participant.api_id=ANY(%s)
        """
        self._insertquery = """
            INSERT INTO participant_stats
            (patch_version, participant_api_id, role)
            VALUES(2.2, %(id)s, %(role)s::json)
            ON CONFLICT(participant_api_id) DO
            UPDATE SET role=%(role)s::json
        """

        self._dimension = 5
        self._bucketsize = 100
        self._hidden = [3]

    def guess_label(self, data):
        hero_map = {
            '*Adagio*': 'captain',
            '*Alpha*': 'jungler',
            '*Ardan*': 'captain',
            '*Baron*': 'carry',
            '*Blackfeather*': 'jungler',
            '*Catherine*': 'captain',
            '*Celeste*': 'carry',
            '*Flicker*': 'captain',
            '*Fortress*': 'captain',
            '*Glaive*': 'jungler',
            '*Grumpjaw*': 'jungler',
            '*Gwen*': 'carry',
            '*Idris*': 'jungler',
            '*Joule*': 'jungler',
            '*Kestrel*': 'carry',
            '*Koshka*': 'jungler',
            '*Hero009*': 'jungler',
            '*Lance*': 'captain',
            '*Lyra*': 'captain',
            '*Ozo*': 'jungler',
            '*Petal*': 'jungler',
            '*Phinn*': 'captain',
            '*Reim*': 'jungler',
            '*Ringo*': 'carry',
            '*Hero016*': 'jungler',
            '*Samuel*': 'carry',
            '*SAW*': 'carry',
            '*Hero010*': 'carry',
            '*Sayoc*': 'jungler',
            '*Skye*': 'carry',
            '*Taka*': 'jungler',
            '*Vox*': 'carry'
        }

        score = {"carry": 0, "jungler": 0, "captain": 0}
#        score["carry"] += data["lanefarm"] / 100  # TODO average lane cs
#        score["jungler"] += data["junglefarm"] / 80  # TODO avg jungle cs
        score[hero_map[data["hero"]]] += 1
        return self._classes.index(max(score, key=score.get))
