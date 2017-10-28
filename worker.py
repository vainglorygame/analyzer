#!/usr/bin/python3
import os
import sys
import time
import logging

from sqlalchemy.orm import sessionmaker, relationship, subqueryload, load_only, selectinload
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.exc import OperationalError
from sqlalchemy import create_engine

import pika
import trueskill
import mpmath
mpmath.mp.dps = 50  # increase if FloatingPointError


RABBITMQ_URI = os.environ.get("RABBITMQ_URI") or "amqp://localhost"
DATABASE_URI = os.environ["DATABASE_URI"]
BATCHSIZE = int(os.environ.get("BATCHSIZE") or 500)  # matches
CHUNKSIZE = int(os.environ.get("CHUNKSIZE") or 100)  # matches
IDLE_TIMEOUT = float(os.environ.get("IDLE_TIMEOUT") or 1)  # s
QUEUE = os.environ.get("QUEUE") or "analyze"
DOCRUNCHMATCH = os.environ.get("DOCRUNCHMATCH") == "true"
CRUNCH_QUEUE = os.environ.get("CRUNCH_QUEUE") or "crunch_global"
DOTELESUCKMATCH = os.environ.get("DOTELESUCKMATCH") == "true"
TELESUCK_QUEUE = os.environ.get("TELESUCK_QUEUE") or "telesuck"
DOSEWMATCH = os.environ.get("DOSEWMATCH") == "true"
SEW_QUEUE = os.environ.get("SEW_QUEUE") or "sew"
UNKNOWN_PLAYER_SIGMA = int(os.environ.get("UNKNOWN_PLAYER_SIGMA") or 500)
TAU = float(os.environ.get("TAU") or 1000/100.0)
RANKED_TAU = float(os.environ.get("RANKED_TAU") or TAU)

# mapping from Tier (-1 - 30) to average skill tier points
vst_points = {
    -1: 1,
    0: 1
}
for c in range(1, 12):
    vst_points[c] = (109 + 1/11) * (c + 0.5)
for c in range(1, 5):
    vst_points[11 + c] = vst_points[11] + 50 * (c + 0.5)
for c in range(1, 10):
    vst_points[15 + c] = vst_points[15] + (66 + 2/3) * (c + 0.5)
for c in range(1, 4):
    vst_points[24 + c] = vst_points[24] + (133 + 1/3) * (c + 0.5)
for c in range(1, 3):
    vst_points[27 + c] = vst_points[27] + 200 * (c + 0.5)
# ---


# ORM definitions
Match = Asset = Roster = Participant = ParticipantStats = ParticipantItems = Player = None
db = rabbit = channel = None

# batch storage
queue = []
timer = None


def connect():
    global Match, Asset, Roster, Participant, ParticipantStats, ParticipantItems, Player
    global Session, rabbit, channel

    # generate schema from db
    Base = automap_base()
    engine = create_engine(DATABASE_URI, pool_size=1, pool_recycle=3600)
    Session = sessionmaker(bind=engine, autoflush=False)
    Base.prepare(engine, reflect=True)

    # definitions
    # TODO check whether the primaryjoin clause is the best method to do this
    Asset = Base.classes.asset
    Match = Base.classes.match
    Match.rosters = relationship(
        "roster", foreign_keys="roster.match_api_id",
        primaryjoin="and_(match.api_id == roster.match_api_id)")
    Match.participants = relationship(
        "participant", foreign_keys="participant.match_api_id",
        primaryjoin="and_(match.api_id == participant.match_api_id)")
    Roster = Base.classes.roster
    Roster.match = relationship(
        "match", foreign_keys="match.api_id",
        primaryjoin="and_(match.api_id == roster.match_api_id)")
    Roster.participants = relationship(
        "participant", foreign_keys="participant.roster_api_id",
        primaryjoin="and_(roster.api_id == participant.roster_api_id)")
    Participant = Base.classes.participant
    Participant.roster = relationship(
        "roster", foreign_keys="roster.api_id",
        primaryjoin="and_(roster.api_id == participant.roster_api_id)")
    Participant.match = relationship(
        "match", foreign_keys="match.api_id",
        primaryjoin="and_(match.api_id == participant.match_api_id)")
    Participant.player = relationship(
        "player", foreign_keys="player.api_id",
        primaryjoin="and_(player.api_id == participant.player_api_id)")
    ParticipantStats = Base.classes.participant_stats
    Participant.participant_stats = relationship(
        "participant_stats", foreign_keys="participant_stats.participant_api_id",
        primaryjoin="and_(participant_stats.participant_api_id == participant.api_id)")
    ParticipantItems = Base.classes.participant_items
    Participant.participant_items = relationship(
        "participant_items", foreign_keys="participant_items.participant_api_id",
        primaryjoin="and_(participant_items.participant_api_id == participant.api_id)")
    Player = Base.classes.player

    rabbit = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URI))
    channel = rabbit.channel()
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.queue_declare(queue=QUEUE + "_failed", durable=True)
    channel.queue_declare(queue=CRUNCH_QUEUE, durable=True)
    channel.queue_declare(queue=TELESUCK_QUEUE, durable=True)
    channel.basic_qos(prefetch_count=BATCHSIZE)
    channel.basic_consume(newjob, queue=QUEUE)


def newjob(_, method, properties, body):
    global timer, queue
    queue.append((method, properties, body))
    if timer is None:
        timer = rabbit.add_timeout(IDLE_TIMEOUT, try_process)
    if len(queue) == BATCHSIZE:
        try_process()

def try_process():
    global timer, queue
    if timer is not None:
        rabbit.remove_timeout(timer)
        timer = None
    try:
        process()
    except Exception as e:
        logger.error(e)
        for meth, prop, body in queue:
            # move to error queue and NACK
            channel.basic_publish(exchange="",
                                  routing_key=QUEUE+"_failed",
                                  body=body,
                                  properties=prop)
            channel.basic_nack(meth.delivery_tag, requeue=False)
        queue = []
        return

    logger.info("acking batch")
    db = None
    if DOTELESUCKMATCH:
        # ro session to get Telemetry URLs
        db = Session(autocommit=True, autoflush=False)

    for meth, prop, body in queue:
        channel.basic_ack(meth.delivery_tag)

        # notify web
        if prop.headers.get("notify"):
            channel.basic_publish("amq.topic", prop.headers.get("notify"),
                                  "analyze_update")

        if DOCRUNCHMATCH:
            # forward to cruncher_global
            channel.basic_publish(exchange="",
                                  routing_key=CRUNCH_QUEUE,
                                  body=body,
                                  properties=prop)
        if DOSEWMATCH:
            # forward to sewer
            channel.basic_publish(exchange="",
                                  routing_key=SEW_QUEUE,
                                  body=body,
                                  properties=prop)
        if DOTELESUCKMATCH:
            # forward asset url to telesucker
            id = str(body, "utf-8")
            for asset in db.query(Asset).options(\
                        load_only("url", "match_api_id"))\
                    .filter(Asset.match_api_id == id).all():
                channel.basic_publish(exchange="",
                                      routing_key=TELESUCK_QUEUE,
                                      body=asset.url,
                                      properties=pika.BasicProperties(
                                          headers={
                                              "match_api_id": asset.match_api_id
                                          }
                                      ))

    if db is not None:
        db.close()

    queue = []


def process():
    global timer, queue, Session
    logger.info("analyzing batch %s", str(len(queue)))
    ids = list(set([str(id, "utf-8") for _, _, id in queue]))

    db = Session()
    try:
        env = trueskill.TrueSkill(
            backend="mpmath",
            mu=1500,
            sigma=1000,
            beta=10.0/30*3000,
            tau=TAU,
            draw_probability=0
        )
        ranked_env = trueskill.TrueSkill(
            backend="mpmath",
            mu=1500,
            sigma=1000,
            beta=10.0/30*3000,
            tau=RANKED_TAU,
            draw_probability=0
        )
        for match in db.query(Match).options(\
            load_only("api_id", "game_mode")\
            .selectinload(Match.rosters)\
                .load_only("api_id", "match_api_id", "winner")\
            .selectinload(Roster.participants)\
                .load_only("api_id", "match_api_id", "roster_api_id",
                           "player_api_id", "skill_tier", "went_afk")\
                .selectinload(Participant.player)\
                    .load_only("api_id",
                               "trueskill_sigma", "trueskill_mu",
                               "trueskill_ranked_sigma", "trueskill_ranked_mu")\
         ).filter(Match.api_id.in_(ids)).yield_per(CHUNKSIZE):
            matchup = []  # common TS
            matchup_ranked = []  # queue specific TS
            anyAfk = False
            for roster in match.rosters:
                for participant in roster.participants:
                    if participant.went_afk == 1:
                        anyAfk = True
                        break
            if len(match.rosters) < 2 \
                or len(match.rosters[0].participants) < 3 \
                or len(match.rosters[1].participants) < 3:
                logger.error("got an invalid matchup %s", match.api_id)
                match.trueskill_quality = 0
                continue
            if anyAfk:
                logger.info("got an afk matchup %s", match.api_id)
                match.trueskill_quality = 0
                continue

            for roster in match.rosters:
                team = []  # combined
                team_ranked = []  # competitive
                for participant in roster.participants:
                    player = participant.player[0]
                    # no data -> approximate ts by VST
                    sigma = player.trueskill_sigma or UNKNOWN_PLAYER_SIGMA
                    mu = player.trueskill_mu or vst_points[participant.skill_tier] + sigma

                    # store pre match values
                    participant.trueskill_mu = mu
                    participant.trueskill_sigma = sigma

                    team.append(env.create_rating(float(mu), float(sigma)))

                    if match.game_mode == "ranked":
                        # fallback to combined TrueSkill, ranked only TS was introduced in 2.19
                        sigma_ranked = player.trueskill_ranked_sigma or sigma
                        mu_ranked = player.trueskill_ranked_mu or mu

                        participant.participant_items[0].trueskill_ranked_mu = mu_ranked
                        participant.participant_items[0].trueskill_ranked_sigma = sigma_ranked

                        team_ranked.append(ranked_env.create_rating(float(mu_ranked), float(sigma_ranked)))

                matchup.append(team)

                if match.game_mode == "ranked":
                    matchup_ranked.append(team_ranked)

            logger.info("got a valid matchup %s", match.api_id)

            # store the fairness of the match
            match.trueskill_quality = env.quality(matchup)
            for team, roster in zip(env.rate(matchup, ranks=[int(not r.winner) for r in match.rosters]),
                                    match.rosters):
                # lower rank is better = winner!
                for rating, participant in zip(team, roster.participants):
                    player = participant.player[0]
                    player.trueskill_mu = rating.mu
                    player.trueskill_sigma = rating.sigma
                    # delta = current - pre
                    participant.trueskill_delta = (float(player.trueskill_mu) - float(player.trueskill_sigma)) - (float(participant.trueskill_mu) - float(participant.trueskill_sigma))

            if match.game_mode == "ranked":
                for team, roster in zip(ranked_env.rate(matchup_ranked, ranks=[int(not r.winner) for r in match.rosters]),
                                        match.rosters):
                    for rating, participant in zip(team, roster.participants):
                        player = participant.player[0]
                        player.trueskill_ranked_mu = rating.mu
                        player.trueskill_ranked_sigma = rating.sigma

        db.commit()
    except:
        db.rollback()
        raise
    finally:
        db.close()


# log errs to stderr, debug to stdout
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno in (logging.DEBUG, logging.INFO)

logger = logging.getLogger("__name__")
logger.setLevel(logging.INFO)

h1 = logging.StreamHandler(sys.stdout)
h1.setLevel(logging.INFO)
h1.addFilter(InfoFilter())
logger.addHandler(h1)

h2 = logging.StreamHandler(sys.stderr)
h2.setLevel(logging.WARNING)
logger.addHandler(h2)

if __name__ == "__main__":
    connect()
    channel.start_consuming()
