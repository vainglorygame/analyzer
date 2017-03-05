#!/usr/bin/python3

import os
import argparse
import asyncio
import asyncpg

import joblib.joblib

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


async def main(qdb, sdb, name):
    queue = joblib.joblib.JobQueue()
    await queue.connect(**qdb)
    await queue.setup()
    pool = await asyncpg.create_pool(**sdb)

    async with pool.acquire() as con:
        async with con.transaction():
            participants = await con.fetch("""
select
unnest(array[
roster.participant_1,
roster.participant_2,
roster.participant_3
]) AS api_id
from roster where roster.match_api_id in (
select
match.api_id
from player
join participant on participant.player_api_id=player.api_id
join roster on participant.roster_api_id=roster.api_id
join match on roster.match_api_id=match.api_id
where player.name=$1
)
            """, name)
    payload = [{
        "id": part["api_id"],
        "type": "participant"
    } for part in participants]
    await queue.request(jobtype="analyze",
                        payload=payload)

parser = argparse.ArgumentParser(description="Request a Vainsocial analyze.")
parser.add_argument("-n", "--player",
                    help="Player name",
                    type=str)
args = parser.parse_args()

loop = asyncio.get_event_loop()
loop.run_until_complete(main(queue_db, db_config, args.player))
