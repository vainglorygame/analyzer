#!/usr/bin/python

import os
import itertools
import json
import asyncio
import asyncpg
import logging
import numpy as np
import tensorflow as tf
import psycopg2

import joblib.worker

#tf.logging.set_verbosity(tf.logging.WARNING)

queue_db = {
    "host": os.environ.get("POSTGRESQL_SOURCE_HOST") or "localhost",
    "port": os.environ.get("POSTGRESQL_SOURCE_PORT") or 5433,
    "user": os.environ.get("POSTGRESQL_SOURCE_USER") or "vainraw",
    "password": os.environ.get("POSTGRESQL_SOURCE_PASSWORD") or "vainraw",
    "database": os.environ.get("POSTGRESQL_SOURCE_DB") or "vainsocial-raw"
}

db_config = {
    "host": os.environ.get("POSTGRESQL_DEST_HOST") or "localhost",
    "port": os.environ.get("POSTGRESQL_DEST_PORT") or 5432,
    "user": os.environ.get("POSTGRESQL_DEST_USER") or "vainweb",
    "password": os.environ.get("POSTGRESQL_DEST_PASSWORD") or "vainweb",
    "database": os.environ.get("POSTGRESQL_DEST_DB") or "vainsocial-web"
}


# TODO create abstract class, move to own file
class KDAClassifier(object):
    """A DNN that classifies loss/win based on KDA."""
    def __init__(self):
        self.model = None
        self.modeldir = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))) + "/models/kda-win"
        self._pool = None

        # DNN configuration
        self._categories = []
        self._continuous = ["kills", "deaths", "assists"]
        self._classes = ["loss", "win"]
        self._label = "win"

        # learn configuration
        self._steps = 1000  # per batch
        self._batchsize = 500
        self._max_batches = 20  # number of batches to train

        # database mappings
        # TODO cache, rm duplicated code
        self._trainquery = """
            SELECT
                kills/duration::float,
                deaths/duration::float,
                assists/duration::float,
                participant.winner,
                participant.api_id
            FROM participant
            TABLESAMPLE BERNOULLI(5)
            JOIN roster on participant.roster_api_id=roster.api_id
            JOIN match on roster.match_api_id=match.api_id
            LIMIT %s
        """
        self._testquery = """
            SELECT
                kills/duration::float,
                deaths/duration::float,
                assists/duration::float,
                participant.winner,
                participant.api_id
            FROM participant
            JOIN roster on participant.roster_api_id=roster.api_id
            JOIN match on roster.match_api_id=match.api_id
            WHERE participant.api_id=ANY(%s)
        """

    def connect(self, **args):
        self._conn = psycopg2.connect(**args)

    def _get_sample(self, size=None, filter=None):
        """Return a data set from the database.
        :param size: (optional) The number of items to get. Defaults to `All`.
        :type size: int or str
        :param offset: (optional) The SQL OFFSET parameter.
        :type offset: int or str
        """
        cur = self._conn.cursor()
        assert (filter is None) ^ (size is None)
        if filter is None:
            cur.execute(self._trainquery, (size,))
        else:
            cur.execute(self._testquery, (filter,))
        ids = []
        data = {"kills": [], "deaths": [], "assists": [], "win": []}
        for rec in cur:
            data["kills"].append(rec[0])
            data["deaths"].append(rec[1])
            data["assists"].append(rec[2])
            data["win"].append(rec[3])
            ids.append(rec[4])
        cur.close()
        logging.debug(data)
        return data, ids

    def _model_setup(self):
        """Sets up a model that takes the `num_features` features as input."""
        feature_columns = [
            tf.contrib.layers.sparse_column_with_hash_bucket(feat, 100)
            for feat in self._categories
        ]
        emb_columns = [
            tf.contrib.layers.embedding_column(
                sparse_id_column=col,
                dimension=2  # log_2(number of unique features)  TODO
            )
            for col in feature_columns
        ] + [
            tf.contrib.layers.real_valued_column(feat)
            for feat in self._continuous
        ]

        self.model = tf.contrib.learn.DNNClassifier(
            feature_columns=emb_columns,
            hidden_units=[2],  # http://stats.stackexchange.com/a/1097  TODO inp+outp / 2
            n_classes=len(self._classes),
            model_dir=self.modeldir,
            config=tf.contrib.learn.RunConfig(
                save_checkpoints_secs=2
            )
        )

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

    # TODO use asyncpg cursor https://magicstack.github.io/asyncpg/current/api/index.html#cursors
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
            input_fn=lambda: self._toinput(self._get_sample(size=2000)[0], train=True),
            eval_steps=1,
            every_n_steps=10
        )

        for _ in range(self._max_batches):
            self.model.fit(
                input_fn=lambda: self._more(),
                steps=self._steps,
                monitors=[validation_monitor]
            )

    def classify(self, sample, only_best=False):
        """Classify a data set.

        :param only_best: (optional) Return the predicted result
                          instead of a dict of propabilities.
        :type only_best: bool
        :return: Prediction results.
        :rtype: list of dict or list
        """
        if only_best:
            return self.model.predict(input_fn=lambda: self._toinput(sample, train=False))
        else:
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
        logging.error("sample size: %s", len(objids))
        sample, objids = self._get_sample(filter=objids)
        d = self.classify(sample)
        cnt = 0
        cur = self._conn.cursor()
        for l in itertools.islice(d, len(objids)):
            cur.execute("""
                INSERT INTO participant_stats
                (patch_version, participant_api_id, score)
                VALUES(2.2, %(objid)s, %(score)s)
                ON CONFLICT(participant_api_id) DO
                UPDATE SET score=%(score)s
            """, {"objid": objids[cnt], "score": float(l[1])})
            cnt += 1
        cur.close()


class Analyzer(joblib.worker.Worker):
    def __init__(self):
        self._pool = None
        self._queries = {}
        super().__init__(jobtype="analyze")
        self.classifier = None

    async def connect(self, dbconf, queuedb):
        """Connect to database."""
        logging.warning("connecting to database")
        await super().connect(**queuedb)
        self._pool = await asyncpg.create_pool(**dbconf)
        self.classifier = KDAClassifier()
        self.classifier.connect(**dbconf)

    async def setup(self):
        """Setup the model."""
        self.classifier.train()

    async def _windup(self):
        self._con = await self._pool.acquire()
        self._tr = self._con.transaction()
        await self._tr.start()
        self.classifier.windup()
        self._participants = []

    async def _teardown(self, failed):
        if len(self._participants) > 0:
            # TODO if this fails, job is still marked as finished
            self.classifier.classify_db(self._participants)

        if failed:
            await self._tr.rollback()
        else:
            await self._tr.commit()
        await self._pool.release(self._con)
        self.classifier.teardown()
    
    async def _execute_job(self, jobid, payload, priority):
        object_id = payload["id"]
        object_type = payload["type"]
        if object_type != "participant":
            return
        self._participants.append(object_id)
        logging.info("%s: classifying '%s', %s", jobid,
                     object_type, object_id)

async def startup():
    for _ in range(1):
        worker = Analyzer()
        await worker.connect(db_config, queue_db)
        await worker.setup()
        await worker.start(batchlimit=1000)


logging.basicConfig(level=logging.DEBUG)

loop = asyncio.get_event_loop()
loop.run_until_complete(startup())
loop.run_forever()
