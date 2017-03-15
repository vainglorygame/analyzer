#!/usr/bin/python

import os
import asyncio
import logging

import joblib.worker
import classifier


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


class Analyzer(joblib.worker.Worker):
    def __init__(self):
        self._queries = {}
        super().__init__(jobtype="analyze")
        self.classifiers = []

    async def connect(self, dbconf, queuedb):
        """Connect to database."""
        logging.warning("connecting to database")
        await super().connect(**queuedb)
        cl = classifier.KDAClassifier()
        cl.connect(**dbconf)
        self.classifiers.append(cl)
#        cl = classifier.RoleClassifier()
#        cl.connect(**dbconf)
#        self.classifiers.append(cl)

    async def setup(self):
        """Setup the model."""
        for cl in self.classifiers:
            cl.train()

    async def _windup(self):
        for cl in self.classifiers:
            cl.windup()
        self._participants = []

    async def _teardown(self, failed):
        if len(self._participants) > 0:
            # TODO if this fails, job is still marked as finished
            for cl in self.classifiers:
                cl.classify_db(self._participants)
        for cl in self.classifiers:
            cl.teardown()
    
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
        await worker.start(batchlimit=10000)


logging.basicConfig(level=logging.DEBUG)

loop = asyncio.get_event_loop()
loop.run_until_complete(startup())
loop.run_forever()
