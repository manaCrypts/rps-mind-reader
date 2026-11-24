import numpy as np
from rps_ai.game import beats, outcome, Match, BOTS, CyclicBot, SwitchingBot
from rps_ai.predictors import FrequencyPredictor, MarkovPredictor, HMMPredictor, RandomPredictor
from rps_ai.agent import make_agent, AI_NAMES


def test_rules():
    assert [beats(m) for m in range(3)] == [1, 2, 0]       # P>R, S>P, R>S
    assert outcome(1, 0) == 1 and outcome(0, 1) == -1 and outcome(2, 2) == 0
    assert all(outcome(beats(m), m) == 1 for m in range(3))


def test_bots_are_seeded():
    for name, cls in BOTS.items():
        a = [r["m1"] for r in Match(cls(seed=3), cls(seed=9), 60).run()]
        b = [r["m1"] for r in Match(cls(seed=3), cls(seed=9), 60).run()]
        assert a == b, name


def test_cyclic_and_switching():
    bot = CyclicBot(seed=0, noise=0.0)
    assert [bot.next_move() for _ in range(6)] == [0, 1, 2, 0, 1, 2]
    assert SwitchingBot(seed=1).period == 50


def test_probabilities_valid():
    for p in (RandomPredictor(), FrequencyPredictor(), MarkovPredictor(1), MarkovPredictor(2), HMMPredictor()):
        rng = np.random.default_rng(0)
        for m in rng.integers(0, 3, 60):
            pr = p.predict_proba()
            assert pr.shape == (3,) and abs(pr.sum() - 1) < 1e-9 and (pr >= 0).all()
            p.update(m)


def test_markov_learns_pattern():
    p = MarkovPredictor(1)
    for m in [0, 1, 2] * 10:
        p.update(m)
    assert p.predict_proba().argmax() == 0               # after S comes R


def test_frequency_learns_bias():
    p = FrequencyPredictor()
    for m in [2] * 15:
        p.update(m)
    assert p.predict_proba().argmax() == 2


def test_hmm_runs_and_refits():
    p = HMMPredictor()
    for m in [0, 1, 2] * 20:
        p.update(m)
    assert p.fits > 0 and p.predict_proba().argmax() == 0


def test_ai_beats_random_on_patterns_and_is_fair_vs_random():
    for ai in AI_NAMES[2:]:
        m = Match(make_agent(ai, 0), CyclicBot(seed=0), 150); m.run()
        assert m.score[0] / 150 > 0.5, ai
    m = Match(make_agent("Markov2", 0), BOTS["Random"](seed=0), 600); m.run()
    assert abs(m.score[0] / 600 - 1 / 3) < 0.08


def test_agent_commits_before_seeing_opponent():
    a = make_agent("Markov1", 0)
    mv = a.next_move()                                   # chosen with no knowledge of this round
    assert mv in (0, 1, 2) and a.last_pred == (mv - 1) % 3
