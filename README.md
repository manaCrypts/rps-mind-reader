# Rock-Paper-Scissors Mind Reader
AL-2002 Artificial Intelligence Lab · Fall 2026 · Manahil Shaikh (24K-2020, CY-5A)

An AI that secretly learns how you play rock-paper-scissors and counters your predicted next move.
Predictors: **Frequency** (sliding window), **Markov chains** (order 1 and 2, Laplace smoothing),
**Hidden Markov Model** (hidden state = current strategy; own forward algorithm + `hmmlearn` Baum-Welch refits),
and a **Random** baseline (33.3%).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py                 # play the game (AI commits its move before you choose)
python run_experiments.py            # full tournament: 30 seeds x 300 rounds x 40 matchups (~1 min) + 4 charts
python run_experiments.py --cached   # redraw charts from data/*.csv (no re-run)
pytest -q                            # 9 unit tests
```

## Layout
| File | Content |
|---|---|
| `rps_ai/game.py` | rules (`beats`, `outcome`), `Player`, 6 seeded `ScriptedBot`s, `HumanPlayer`, `Match` |
| `rps_ai/predictors.py` | `Predictor` (abstract), `RandomPredictor`, `FrequencyPredictor`, `MarkovPredictor`, `HMMPredictor` |
| `rps_ai/agent.py` | `Agent` has-a `Predictor`; plays the counter to the likeliest move |
| `rps_ai/tournament.py` | round-robin: 5 AIs x 6 bots + 10 predictor pairs, seeds 0-29 |
| `run_experiments.py` | runs tournament, writes `data/*.csv`, `results/*.csv`, `results/chart1-4*.png` |
| `app.py` | Streamlit game with live rolling win-rate chart; can log human sessions to `data/human_sessions.csv` |
| `tests/test_rps.py` | rules, seeding, predictor validity/learning, HMM refit, fairness vs random |

## Results (30 seeds, 300 rounds; win = AI wins the round, luck = 33.3%)
| AI \ Bot | Random | Biased | Cyclic | WSLS | CounterLast | Switching |
|---|---|---|---|---|---|---|
| Random | 0.334 | 0.334 | 0.333 | 0.333 | 0.340 | 0.333 |
| Frequency | 0.333 | 0.464 | 0.106 | 0.135 | 0.131 | 0.423 |
| Markov1 | 0.339 | 0.480 | 0.924 | 0.745 | 0.356 | 0.478 |
| Markov2 | 0.342 | 0.449 | 0.917 | 0.908 | 0.508 | 0.556 |
| HMM | 0.334 | 0.473 | 0.904 | 0.836 | 0.405 | 0.457 |

| AI | mean win-rate (= prediction accuracy) | rounds to pass 40% (rolling 20) | ms / move |
|---|---|---|---|
| Random | 0.335 | 51 | 0.02 |
| Frequency | 0.265 | 167 | 0.02 |
| Markov1 | 0.554 | 35 | 0.02 |
| Markov2 | **0.613** | **33** | 0.02 |
| HMM | 0.568 | 33 | 0.38 |

## Honest findings
* Every predictor stays at about 33% against the Random bot (no free lunch) and clearly beats luck against patterned bots.
* **Markov order 2 is best overall** (61%); it exceeds 50% against Cyclic, WSLS, CounterLast and Switching.
* The proposal expected the HMM to do best on the Switching bot. It did **not**: Markov2 (0.556) beat HMM (0.457). Likely cause: with
  only 3 states and refits every 10 rounds on short histories, the HMM adapts to a strategy change more slowly than the
  count-based Markov table does (Risk 1 of the proposal). It does learn cycles well (0.90 vs Cyclic) and is the slowest predictor.
* The plain Frequency predictor is *exploitable*: Cyclic, WSLS and CounterLast bots beat it (win-rate 11-14%), because it
  always plays the counter to the most common move, which those bots anticipate.
* Win-rate equals prediction accuracy by construction: the AI wins exactly when its predicted move is correct (a tie/loss otherwise
  is counted as "not a win"; 33.3% is the luck level).
* Human playtests: use the app's "Save session" button to log classmates' games to `data/human_sessions.csv` (not included - no logged sessions yet).
