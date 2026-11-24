"""Game rules, players (scripted bots, human) and the Match loop."""
import random
import time
from abc import ABC, abstractmethod

MOVES = ["Rock", "Paper", "Scissors"]
ROCK, PAPER, SCISSORS = 0, 1, 2


def beats(move):
    """Return the move that beats `move` (Paper>Rock, Scissors>Paper, Rock>Scissors)."""
    return (move + 1) % 3


def outcome(a, b):
    """Result for the player who played `a` against `b`: +1 win, 0 tie, -1 loss."""
    d = (a - b) % 3
    return 0 if d == 0 else (1 if d == 1 else -1)


class Player(ABC):
    """Abstract player. next_move() is called before the opponent's move is known."""
    name = "player"

    def __init__(self, seed=None):
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng = random.Random(self.seed)
        self.t = 0
        self.last_own = None
        self.last_opp = None
        self._clear()

    def _clear(self):
        pass

    @abstractmethod
    def next_move(self) -> int: ...

    def observe(self, own, opp):
        self.last_own, self.last_opp = own, opp


class ScriptedBot(Player):
    """Base for bots: optional noise makes them behave a little like humans."""
    def __init__(self, seed=None, noise=0.0):
        self.noise = noise
        super().__init__(seed)

    def _noisy(self, m):
        return self.rng.randrange(3) if self.rng.random() < self.noise else m

    def next_move(self):
        m = self._noisy(self._choose())
        self.t += 1
        return m

    @abstractmethod
    def _choose(self) -> int: ...


class RandomBot(ScriptedBot):
    name = "Random"
    def _choose(self): return self.rng.randrange(3)


class BiasedBot(ScriptedBot):
    name = "Biased"
    def __init__(self, seed=None, probs=(0.5, 0.3, 0.2), noise=0.0):
        self.probs = probs
        super().__init__(seed, noise)
    def _choose(self): return self.rng.choices([0, 1, 2], weights=self.probs)[0]


class CyclicBot(ScriptedBot):
    name = "Cyclic"
    def __init__(self, seed=None, noise=0.05): super().__init__(seed, noise)
    def _choose(self): return self.t % 3          # R -> P -> S -> R ...


class WinStayLoseShiftBot(ScriptedBot):
    name = "WSLS"
    def __init__(self, seed=None, noise=0.05): super().__init__(seed, noise)
    def _choose(self):
        if self.last_own is None:
            return self.rng.randrange(3)
        won = outcome(self.last_own, self.last_opp) == 1
        return self.last_own if won else (self.last_own + 1) % 3


class CounterLastBot(ScriptedBot):
    """Plays the move that beats the AI's previous move."""
    name = "CounterLast"
    def __init__(self, seed=None, noise=0.05): super().__init__(seed, noise)
    def _choose(self):
        return self.rng.randrange(3) if self.last_opp is None else beats(self.last_opp)


class SwitchingBot(ScriptedBot):
    """Changes strategy every `period` rounds: rock-heavy -> cyclic -> scissors-heavy."""
    name = "Switching"
    def __init__(self, seed=None, period=50):
        self.period = period
        super().__init__(seed, 0.0)

    def _clear(self):
        s = self.seed
        self.subs = [BiasedBot(s, (0.6, 0.2, 0.2)), CyclicBot(s),
                     BiasedBot(s, (0.2, 0.2, 0.6))]

    def _choose(self):
        return self.subs[(self.t // self.period) % len(self.subs)].next_move()

    def observe(self, own, opp):
        super().observe(own, opp)
        for b in self.subs:
            b.observe(own, opp)


BOTS = {c.name: c for c in (RandomBot, BiasedBot, CyclicBot, WinStayLoseShiftBot,
                            CounterLastBot, SwitchingBot)}


class HumanPlayer(Player):
    """Move comes from the UI (Streamlit button press)."""
    name = "Human"
    def __init__(self): super().__init__(None); self.pending = None
    def set_move(self, m): self.pending = m
    def next_move(self): return self.pending


class Match:
    """Round loop: both players commit, moves are revealed, both observe."""
    def __init__(self, p1, p2, rounds=300):
        self.p1, self.p2, self.rounds = p1, p2, rounds
        self.reset()

    def reset(self):
        self.p1.reset(); self.p2.reset()
        self.rows, self.score = [], [0, 0, 0]  # p1 wins, ties, p1 losses

    def play_round(self):
        t0 = time.perf_counter(); m1 = self.p1.next_move(); t1 = time.perf_counter() - t0
        t0 = time.perf_counter(); m2 = self.p2.next_move(); t2 = time.perf_counter() - t0
        res = outcome(m1, m2)
        t0 = time.perf_counter(); self.p1.observe(m1, m2); t1 += time.perf_counter() - t0
        t0 = time.perf_counter(); self.p2.observe(m2, m1); t2 += time.perf_counter() - t0
        self.score[{1: 0, 0: 1, -1: 2}[res]] += 1
        row = dict(round=len(self.rows) + 1, m1=m1, m2=m2, result=res,
                   pred1=getattr(self.p1, "last_pred", -1), pred2=getattr(self.p2, "last_pred", -1),
                   ms1=t1 * 1000, ms2=t2 * 1000)
        self.rows.append(row)
        return row

    def run(self):
        self.reset()
        for _ in range(self.rounds):
            self.play_round()
        return self.rows
