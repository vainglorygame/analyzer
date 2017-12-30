#!/usr/bin/python

import pytest
import rater

class Match:
    def __init__(self, game_mode, rosters):
        self.api_id = ""
        self.game_mode = game_mode
        self.rosters = rosters
        self.participants = [p for r in rosters for p in r.participants]

class Roster:
    def __init__(self, winner, participants):
        self.api_id = ""
        self.winner = winner
        self.participants = participants

class Participant:
    def __init__(self, skill_tier, went_afk,
                 trueskill_mu, trueskill_sigma,
                 participant_items, player):
        self.api_id = ""
        self.skill_tier = skill_tier
        self.went_afk = went_afk
        self.participant_items = [participant_items]
        self.player = [player]

class ParticipantItems:
    def __init__(self, trueskill_casual_mu, trueskill_casual_sigma,
                 trueskill_ranked_mu, trueskill_ranked_sigma,
                 trueskill_blitz_mu, trueskill_blitz_sigma,
                 trueskill_br_mu, trueskill_br_sigma):
        self.api_id = ""
        self.trueskill_casual_mu = trueskill_casual_mu
        self.trueskill_casual_sigma = trueskill_casual_sigma
        self.trueskill_ranked_mu = trueskill_ranked_mu
        self.trueskill_ranked_sigma = trueskill_ranked_sigma
        self.trueskill_blitz_mu = trueskill_blitz_mu
        self.trueskill_blitz_sigma = trueskill_blitz_sigma
        self.any_afk = False

class Player:
    def __init__(self, skill_tier,
                 rank_points_ranked,
                 rank_points_blitz,
                 trueskill_mu, trueskill_sigma,
                 trueskill_casual_mu, trueskill_casual_sigma,
                 trueskill_ranked_mu, trueskill_ranked_sigma,
                 trueskill_blitz_mu, trueskill_blitz_sigma,
                 trueskill_br_mu, trueskill_br_sigma):
        self.api_id = ""
        self.skill_tier = skill_tier
        self.rank_points_ranked = rank_points_ranked
        self.rank_points_blitz = rank_points_blitz
        self.trueskill_mu = trueskill_mu
        self.trueskill_sigma = trueskill_sigma
        self.trueskill_casual_mu = trueskill_casual_mu
        self.trueskill_casual_sigma = trueskill_casual_sigma
        self.trueskill_ranked_mu = trueskill_ranked_mu
        self.trueskill_ranked_sigma = trueskill_ranked_sigma
        self.trueskill_blitz_mu = trueskill_blitz_mu
        self.trueskill_blitz_sigma = trueskill_blitz_sigma


class TestRater:
    def test_get_trueskill_seed(self):
        # test approx. by skill tier
        player = Player(15, None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None)
        mu, sigma = rater.get_trueskill_seed(player)
        assert 1300 < mu - sigma < 1700

        # test approx. by rank points
        player = Player(0, 2500, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None)
        mu, sigma = rater.get_trueskill_seed(player)
        assert mu - sigma == 2500

        player = Player(0, 2500, 100,
                        None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None)
        mu, sigma = rater.get_trueskill_seed(player)
        assert mu - sigma == 2500

        player = Player(0, 100, 2500,
                        None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None)
        mu, sigma = rater.get_trueskill_seed(player)
        assert mu - sigma == 2500

        player = Player(0, None, 2500,
                        None, None,
                        None, None,
                        None, None,
                        None, None,
                        None, None)
        mu, sigma = rater.get_trueskill_seed(player)
        assert mu - sigma == 2500

    def test_rate_match(self):
        # new user, only skill tier
        g_player = lambda: Player(15, None, None,
                                None, None,
                                None, None,
                                None, None,
                                None, None,
                                None, None)
        g_participant_items = lambda: ParticipantItems(None, None,
                                                     None, None,
                                                     None, None,
                                                     None, None)
        g_participant = lambda: Participant(0, False,
                                          None, None,
                                          g_participant_items(), g_player())
        roster = Roster(True, [g_participant()]*3)
        roster2 = Roster(False, [g_participant()]*3)
        match = Match("ranked", [roster, roster2])

        rater.rate_match(match)

        assert match.rosters[0].participants[0].player[0].trueskill_mu is not None
        assert match.rosters[0].participants[0].player[0].trueskill_ranked_mu is not None
        assert match.rosters[0].participants[0].player[0].trueskill_ranked_sigma < match.rosters[0].participants[0].player[0].trueskill_ranked_mu
        assert 500 < match.rosters[0].participants[0].player[0].trueskill_ranked_mu < 2500
        assert match.rosters[0].participants[0].player[0].trueskill_casual_mu is None
        assert match.rosters[0].participants[0].player[0].trueskill_mu > match.rosters[1].participants[0].player[0].trueskill_mu
        assert match.rosters[0].participants[0].player[0].trueskill_ranked_mu > match.rosters[1].participants[0].player[0].trueskill_ranked_mu

    def test_rate_match_returning(self):
        # returning user
        g_player = lambda: Player(None, None, None,
                                  2000, 100,
                                  None, None,
                                  None, None,
                                  None, None,
                                  None, None)
        g_participant_items = lambda: ParticipantItems(None, None,
                                                       None, None,
                                                       None, None,
                                                       None, None)
        g_participant = lambda: Participant(0, False,
                                            None, None,
                                            g_participant_items(), g_player())
        roster = Roster(True, [g_participant()]*3)
        roster2 = Roster(False, [g_participant()]*3)
        match = Match("ranked", [roster, roster2])

        rater.rate_match(match)

        assert 1800 < match.rosters[0].participants[0].player[0].trueskill_ranked_mu < 2200

    def test_rate_match_afk(self):
        # afk
        g_player = lambda: Player(None, None, None,
                                  None, None,
                                  None, None,
                                  None, None,
                                  None, None,
                                  None, None)
        g_participant_items = lambda: ParticipantItems(None, None,
                                                       None, None,
                                                       None, None,
                                                       None, None)
        g_participant = lambda: Participant(0, True,
                                            None, None,
                                            g_participant_items(), g_player())
        roster = Roster(True, [g_participant()]*3)
        roster2 = Roster(False, [g_participant()]*3)
        match = Match("ranked", [roster, roster2])

        rater.rate_match(match)

        assert match.rosters[0].participants[0].player[0].trueskill_mu is None
        assert match.rosters[0].participants[0].participant_items[0].any_afk == True
