"""Tournament: every predictor vs every scripted bot, and predictor vs predictor."""
from itertools import combinations
import numpy as np
import pandas as pd
from .game import BOTS, Match
from .agent import AI_NAMES, make_agent

ROLL = 20


def rounds_to_threshold(results, thr=0.40, win=ROLL):
    """First round at which the rolling win-rate (window `win`) exceeds thr (cap = len)."""
    w = (np.asarray(results) == 1).astype(float)
    if len(w) < win:
        return len(w)
    roll = np.convolve(w, np.ones(win) / win, mode="valid")
    idx = np.flatnonzero(roll > thr)
    return int(idx[0] + win) if len(idx) else len(w)


def run_one(p1, p2, seed, rounds=300, kind="bot"):
    """Play one match. p2 is a bot name or an AI name; returns (summary dict, results array)."""
    a = make_agent(p1, seed)
    b = BOTS[p2](seed=seed) if kind == "bot" else make_agent(p2, seed + 500)
    m = Match(a, b, rounds); rows = m.run()
    res = np.array([r["result"] for r in rows])
    m1 = np.array([r["m2"] for r in rows]); p1p = np.array([r["pred1"] for r in rows])
    s = dict(p1=p1, p2=p2, seed=seed, kind=kind,
             win_rate=(res == 1).mean(), loss_rate=(res == -1).mean(), tie_rate=(res == 0).mean(),
             pred_acc=(p1p == m1).mean(), ms_per_move=np.mean([r["ms1"] for r in rows]),
             rounds_to_40=rounds_to_threshold(res))
    return s, res


def _job(args):
    return run_one(*args)


def matchups():
    bots = [(a, b) for a in AI_NAMES for b in BOTS]
    pairs = [(a, b) for a, b in combinations(AI_NAMES, 2)]
    return bots, pairs


def run_tournament(seeds=range(30), rounds=300):
    bots, pairs = matchups()
    jobs = [(a, b, s, rounds, k) for k, lst in (("bot", bots), ("pair", pairs))
            for a, b in lst for s in seeds]
    out = [_job(j) for j in jobs]
    summ = pd.DataFrame([o[0] for o in out])
    rec = []
    for (s, res) in out:
        rec.append(pd.DataFrame({"p1": s["p1"], "p2": s["p2"], "seed": s["seed"],
                                 "round": np.arange(1, len(res) + 1), "result": res.astype(np.int8)}))
    return summ, pd.concat(rec, ignore_index=True)
