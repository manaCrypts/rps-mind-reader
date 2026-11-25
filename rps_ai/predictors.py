"""Predictors of the opponent's next move. Each outputs P(Rock), P(Paper), P(Scissors)."""
import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np

logging.getLogger("hmmlearn").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")
UNIFORM = np.full(3, 1 / 3)


class Predictor(ABC):
    name = "predictor"

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []

    def update(self, move):
        self.history.append(int(move))

    @abstractmethod
    def predict_proba(self) -> np.ndarray: ...


class RandomPredictor(Predictor):
    """Baseline: always uniform -> the agent plays uniformly at random (33.3%)."""
    name = "Random"
    def predict_proba(self): return UNIFORM.copy()


class FrequencyPredictor(Predictor):
    """Counts the opponent's moves in a sliding window (Laplace smoothed)."""
    name = "Frequency"
    def __init__(self, window=20, alpha=1.0):
        self.window, self.alpha = window, alpha
        super().__init__()

    def predict_proba(self):
        c = np.bincount(self.history[-self.window:], minlength=3) + self.alpha
        return c / c.sum()


class MarkovPredictor(Predictor):
    """Order-k Markov chain: P(next | last k moves) with Laplace smoothing."""
    def __init__(self, order=1, alpha=0.5):
        self.order, self.alpha = order, alpha
        self.name = f"Markov{order}"
        super().__init__()

    def reset(self):
        super().reset()
        self.table = {}                       # context tuple -> counts[3]

    def update(self, move):
        if len(self.history) >= self.order:
            ctx = tuple(self.history[-self.order:])
            self.table.setdefault(ctx, np.zeros(3))[int(move)] += 1
        super().update(move)

    def predict_proba(self):
        if len(self.history) < self.order:
            return UNIFORM.copy()
        c = self.table.get(tuple(self.history[-self.order:]), np.zeros(3)) + self.alpha
        return c / c.sum()


class HMMPredictor(Predictor):
    """Hidden state = the player's current strategy; observation = move.
    Online forward algorithm (own code) + periodic Baum-Welch refit (hmmlearn)."""
    name = "HMM"

    def __init__(self, n_states=3, refit_every=10, window=100, min_fit=20):
        self.n, self.refit_every, self.window, self.min_fit = n_states, refit_every, window, min_fit
        super().__init__()

    def reset(self):
        super().reset()
        n = self.n
        self.pi = np.full(n, 1 / n)
        # sticky strategies with a slight preference for "next" state (helps learn cycles)
        A = np.full((n, n), 0.1)
        for i in range(n):
            A[i, i] = 0.6
            A[i, (i + 1) % n] = 0.3
        self.A = A / A.sum(1, keepdims=True)
        # break symmetry: state i starts out favouring move i
        B = np.full((n, 3), 0.2)
        for i in range(n):
            B[i, i % 3] = 0.6
        self.B = B / B.sum(1, keepdims=True)
        self.alpha = self.pi.copy()
        self.fits = 0

    def _forward(self, move, first):
        a = (self.pi if first else self.alpha @ self.A) * self.B[:, move]
        s = a.sum()
        self.alpha = a / s if s > 0 else np.full(self.n, 1 / self.n)

    def update(self, move):
        super().update(move)
        self._forward(int(move), first=len(self.history) == 1)
        L = len(self.history)
        if L >= self.min_fit and L % self.refit_every == 0:
            self._refit()

    def _refit(self):
        try:
            from hmmlearn.hmm import CategoricalHMM
            obs = np.array(self.history[-self.window:]).reshape(-1, 1)
            m = CategoricalHMM(n_components=self.n, n_features=3, n_iter=15, tol=1e-3,
                               init_params="", params="ste", random_state=0)
            m.startprob_, m.transmat_, m.emissionprob_ = self.pi, self.A, self.B
            m.fit(obs)
            pi = np.clip(m.startprob_, 1e-3, None); A = np.clip(m.transmat_, 1e-3, None)
            B = np.clip(m.emissionprob_, 1e-2, None)
            if not (np.isfinite(pi).all() and np.isfinite(A).all() and np.isfinite(B).all()):
                return
            self.pi, self.A, self.B = pi / pi.sum(), A / A.sum(1, keepdims=True), B / B.sum(1, keepdims=True)
            self.fits += 1
            self.alpha = self.pi.copy()          # re-filter over the fitting window
            for i, mv in enumerate(self.history[-self.window:]):
                self._forward(mv, first=(i == 0))
        except Exception:                        # unstable fit -> keep the old model
            pass

    def predict_proba(self):
        p = (self.alpha @ self.A if self.history else self.pi @ self.A) @ self.B
        return p / p.sum()
