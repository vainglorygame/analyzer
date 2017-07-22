#!/usr/bin/python3
import os
import time
import logging

from sqlalchemy.orm import Session, relationship, subqueryload, load_only, selectinload
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
Match = Roster = Participant = ParticipantStats = Player = None
db = rabbit = channel = None

# batch storage
queue = []
timer = None


def connect():
    global Match, Roster, Participant, ParticipantStats, Player
    global db, rabbit, channel

    # generate schema from db
    Base = automap_base()
    engine = create_engine(DATABASE_URI, pool_size=1, pool_recycle=3600)
    Base.prepare(engine, reflect=True)

    # definitions
    # TODO check whether the primaryjoin clause is the best method to do this
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
    Player = Base.classes.player

    db = Session(engine)

    rabbit = pika.BlockingConnection(pika.URLParameters(RABBITMQ_URI))
    channel = rabbit.channel()
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.queue_declare(queue=QUEUE + "_failed", durable=True)
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
    except e:
        logging.error(e)
        for q in queue:
            # move to error queue and NACK
            channel.basic_publish("", QUEUE + "_failed", q[1], q[2])
            channel.basic_nack(q[0].delivery_tag, requeue=False)
        queue = []
        return

    logging.info("acking batch")
    for q in queue:
        channel.basic_ack(q[0].delivery_tag)
    queue = []


def process():
    global timer, queue, db
    logging.info("analyzing batch %s", str(len(queue)))
    ids = list(set([str(id, "utf-8") for _, _, id in queue]))

    with db.no_autoflush:
        env = trueskill.TrueSkill(
            backend="mpmath",
            mu=1500,
            sigma=1000,
            beta=10.0/30*3000,
            tau=1000/100,
            draw_probability=0
        )
        for match in db.query(Match).options(\
            load_only("api_id")\
            .selectinload(Match.rosters)\
                .load_only("api_id", "match_api_id", "winner")\
            .selectinload(Roster.participants)\
                .load_only("api_id", "match_api_id", "roster_api_id",
                           "player_api_id", "skill_tier", "went_afk",
                           "trueskill_sigma", "trueskill_mu")\
                .selectinload(Participant.player)\
                    .load_only("api_id", "trueskill_sigma", "trueskill_mu")\
         ).filter(Match.api_id.in_(ids)).yield_per(CHUNKSIZE):
            matchup = []
            anyAfk = False
            for roster in match.rosters:
                for participant in roster.participants:
                    if participant.went_afk == 1:
                        anyAfk = True
                        break
            if len(match.rosters) < 2 \
                or len(match.rosters[0].participants) < 3 \
                or len(match.rosters[1].participants) < 3:
                logging.error("got an invalid matchup %s", match.api_id)
                match.trueskill_quality = 0
                continue
            if anyAfk:
                logging.error("got an afk matchup %s", match.api_id)
                match.trueskill_quality = 0
                continue

            for roster in match.rosters:
                team = []
                for participant in roster.participants:
                    player = participant.player[0]
                    mu = participant.trueskill_mu or player.trueskill_mu
                    sigma = participant.trueskill_sigma or player.trueskill_sigma or 500
                    if mu is None:
                        # no data -> approximate ts by VST
                        mu = vst_points[participant.skill_tier] + sigma
                        player.trueskill_mu = mu
                        player.trueskill_sigma = sigma
                    # store pre match values
                    participant.trueskill_mu = mu
                    participant.trueskill_sigma = sigma

                    team.append(env.create_rating(float(mu), float(sigma)))
                matchup.append(team)

            logging.info("got a valid matchup %s", match.api_id)

            # store the fairness of the match
            match.trueskill_quality = env.quality(matchup)
            for team, roster in zip(env.rate(matchup, ranks=[int(not r.winner) for r in match.rosters]),
                                    match.rosters):
                # lower rank is better = winner!
                for rating, participant in zip(team, roster.participants):
                    player = participant.player[0]
                    if player.trueskill_mu == participant.trueskill_mu \
                       and player.trueskill_sigma == participant.trueskill_sigma:
                        # match hasn't been rated before
                        player.trueskill_mu = rating.mu
                        player.trueskill_sigma = rating.sigma
                    participant.trueskill_delta = (participant.mu - participant.sigma) - (float(player.trueskill_mu) - float(player.trueskill_sigma))

    db.commit()
    # notify web
    for api_id in ids:
        channel.basic_publish("amq.topic", "participant." + api_id,
                              "stats_update")


logging.basicConfig(level=logging.INFO)
if __name__ == "__main__":
    connect()
    channel.start_consuming()
