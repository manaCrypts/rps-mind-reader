"""The AI player: has-a Predictor, plays the counter to the likeliest opponent move."""
import numpy as np
from .game import Player, beats
from .predictors import (RandomPredictor, FrequencyPredictor, MarkovPredictor, HMMPredictor)

AI_NAMES = ["Random", "Frequency", "Markov1", "Markov2", "HMM"]


def make_predictor(name):
    return {"Random": RandomPredictor, "Frequency": FrequencyPredictor,
            "Markov1": lambda: MarkovPredictor(1), "Markov2": lambda: MarkovPredictor(2),
            "HMM": HMMPredictor}[name]()


class Agent(Player):
    def __init__(self, predictor, seed=None):
        self.predictor = predictor
        self.name = predictor.name
        super().__init__(seed)

    def _clear(self):
        self.predictor.reset()
        self.last_pred = -1
        self.last_probs = np.full(3, 1 / 3)

    def next_move(self):
        p = self.predictor.predict_proba()
        best = np.flatnonzero(np.isclose(p, p.max()))
        self.last_probs = p
        self.last_pred = int(self.rng.choice(list(best)))   # random tie-break
        return beats(self.last_pred)                         # counter the likeliest move

    def observe(self, own, opp):
        super().observe(own, opp)
        self.predictor.update(opp)


def make_agent(name, seed=None):
    return Agent(make_predictor(name), seed)
