#!/usr/bin/python3
import os
import sys
import logging

import trueskill
import mpmath
mpmath.mp.dps = 50  # increase if FloatingPointError

UNKNOWN_PLAYER_SIGMA = int(os.environ.get("UNKNOWN_PLAYER_SIGMA") or 500)
TAU = float(os.environ.get("TAU") or 1000/100.0)

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

env = trueskill.TrueSkill(
    backend="mpmath",
    mu=1500,
    sigma=1000,
    beta=10.0/30*3000,
    tau=TAU,
    draw_probability=0
)

"""
Return a (mu, sigma) based on information known about a player.
"""
def get_trueskill_seed(player):
    # fallback 1 - approximate by max(ranked, blitz) points
    rank_points = None
    if player.rank_points_ranked is not None and player.rank_points_ranked != 0:
        rank_points = player.rank_points_ranked
    if player.rank_points_blitz is not None and player.rank_points_blitz != 0:
        if rank_points is None:
            rank_points = player.rank_points_blitz
        else:
            if player.rank_points_blitz > rank_points:
                rank_points = player.rank_points_blitz

    if rank_points is not None:
        sigma = UNKNOWN_PLAYER_SIGMA * (2.0/3.0)  # more accurate than skill tier = more trust
        mu = float(rank_points) + sigma
    else:
        # fallback 2 - approximate rank points by skill tier
        sigma = UNKNOWN_PLAYER_SIGMA
        mu = vst_points[player.skill_tier] + sigma

    return (mu, sigma)


"""
Mutate a match structure by updating TrueSkill values.
Returns changed match object.
"""
def rate_match(match):
    trueskill_column = None  # on player
    if match.game_mode == "casual":  # mode names mapped by processor!!!
        trueskill_column = "trueskill_casual"
    if match.game_mode == "ranked":
        trueskill_column = "trueskill_ranked"
    if match.game_mode == "blitz":
        trueskill_column = "trueskill_blitz"
    if match.game_mode == "br":
        trueskill_column = "trueskill_br"
    if match.game_mode == "5v5_casual":
        trueskill_column = "trueskill_5v5_casual"
    if match.game_mode == "5v5_ranked":
        trueskill_column = "trueskill_5v5_ranked"
    if trueskill_column is None:
        logger.info("got unsupported game mode %s", match.game_mode)
        return

    matchup_shared = []  # trueskill shared across all modes
    matchup = []

    anyAfk = False
    if len(match.rosters) != 2:
        logger.error("got an invalid matchup %s", match.api_id)
        anyAfk = True

    for participant in match.participants:
        participant.participant_items[0].any_afk = False
        if participant.went_afk == 1:
            logger.info("got an afk matchup %s", match.api_id)
            anyAfk = True
            break

    if anyAfk:
        match.trueskill_quality = 0
        for participant in match.participants:
            participant.participant_items[0].any_afk = True
        return

    for roster in match.rosters:
        team_shared = []
        team = []
        for participant in roster.participants:
            player = participant.player[0]

            # calculate TrueSkill shared across all modes = starting point for all modes
            if player.trueskill_mu is not None:
                mu_shared = player.trueskill_mu
                sigma_shared = player.trueskill_sigma
            else:
                mu_shared, sigma_shared = get_trueskill_seed(player)

            team_shared.append(env.create_rating(float(mu_shared), float(sigma_shared)))

            # calculate queue specific TrueSkill
            if getattr(player, trueskill_column + "_mu") is not None:
                sigma = getattr(player, trueskill_column + "_sigma")
                mu = getattr(player, trueskill_column + "_mu")
            else:
                # fallback 1 - approximate by shared trueskill
                sigma = sigma_shared
                mu = mu_shared

            team.append(env.create_rating(float(mu), float(sigma)))


        matchup_shared.append(team_shared)
        matchup.append(team)

    logger.info("got a valid matchup %s", match.api_id)

    # store the fairness of the match using the shared TrueSkill
    match.trueskill_quality = env.quality(matchup)

    # shared TrueSkill (participant)
    for team, roster in zip(env.rate(matchup_shared, ranks=[int(not r.winner) for r in match.rosters]),
            match.rosters):
        # lower rank is better = winner!
        for rating, participant in zip(team, roster.participants):
            player = participant.player[0]
            # delta = current - pre
            if player.trueskill_mu is not None:
                participant.trueskill_delta = (float(rating.mu) - float(rating.sigma)) - (float(player.trueskill_mu) - float(player.trueskill_sigma))
            else:
                participant.trueskill_delta = 0
            player.trueskill_mu = rating.mu
            participant.trueskill_mu = rating.mu
            player.trueskill_sigma = rating.sigma
            participant.trueskill_sigma = rating.sigma

    # queue specific TrueSkill (participant_items)
    # no delta
    for team, roster in zip(env.rate(matchup, ranks=[int(not r.winner) for r in match.rosters]),
            match.rosters):
        for rating, participant in zip(team, roster.participants):
            player = participant.player[0]
            pi = participant.participant_items[0]
            setattr(player, trueskill_column + "_mu", rating.mu)
            setattr(pi, trueskill_column + "_mu", rating.mu)
            setattr(player, trueskill_column + "_sigma", rating.sigma)
            setattr(pi, trueskill_column + "_sigma", rating.sigma)


# TODO share this between the two classes
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
