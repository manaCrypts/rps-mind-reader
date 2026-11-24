"""Run the full evaluation (30 seeds x 300 rounds x 40 matchups) and draw the charts.
Usage: python run_experiments.py            (runs the tournament, caches CSVs in data/)
       python run_experiments.py --cached   (redraw charts from data/*.csv only)"""
import sys, time, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rps_ai.tournament import run_tournament, ROLL
from rps_ai.agent import AI_NAMES
from rps_ai.game import BOTS

SUMM, ROUNDS, OUT = "data/tournament_summary.csv", "data/tournament_rounds.csv", "results"
os.makedirs("data", exist_ok=True); os.makedirs(OUT, exist_ok=True)


def load_or_run(cached):
    if cached and os.path.exists(SUMM) and os.path.exists(ROUNDS):
        return pd.read_csv(SUMM), pd.read_csv(ROUNDS)
    t = time.time()
    summ, rounds = run_tournament()
    summ.to_csv(SUMM, index=False); rounds.to_csv(ROUNDS, index=False)
    print(f"tournament finished in {time.time()-t:.0f}s, {len(rounds)} rounds")
    return summ, rounds


def rolling_curve(df):
    """mean rolling win-rate (window ROLL) across seeds for one matchup."""
    piv = df.assign(w=(df.result == 1).astype(float)).pivot(index="round", columns="seed", values="w")
    return piv.rolling(ROLL).mean().mean(axis=1)


def main(cached=False):
    summ, rounds = load_or_run(cached)
    bots = summ[summ.kind == "bot"]; pairs = summ[summ.kind == "pair"]
    bot_names = list(BOTS)

    # ---- tables
    tab = bots.groupby(["p1", "p2"]).win_rate.agg(["mean", "std"]).reset_index()
    wide = tab.pivot(index="p1", columns="p2", values="mean").loc[AI_NAMES, bot_names]
    wide.round(3).to_csv(f"{OUT}/winrate_table.csv")
    metrics = bots.groupby("p1").agg(win_rate=("win_rate", "mean"), pred_acc=("pred_acc", "mean"),
                                     ms_per_move=("ms_per_move", "mean"),
                                     rounds_to_40=("rounds_to_40", "mean")).loc[AI_NAMES]
    metrics.round(3).to_csv(f"{OUT}/metrics_table.csv")
    print(wide.round(3).to_string()); print(metrics.round(3).to_string())

    # ---- chart 1: rolling win-rate vs round, one panel per bot
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    rb = rounds[rounds.p2.isin(bot_names)]
    key = pd.concat([summ[summ.kind == "bot"][["p1", "p2"]].drop_duplicates()])
    for ax, b in zip(axs.flat, bot_names):
        for a in AI_NAMES:
            sub = rb[(rb.p1 == a) & (rb.p2 == b)]
            ax.plot(rolling_curve(sub), label=a, lw=1.6, ls="--" if a == "Random" else "-")
        ax.axhline(1 / 3, color="gray", lw=.8, ls=":"); ax.set_title(f"vs {b} bot"); ax.set_xlabel("round")
    axs[0, 0].set_ylabel("rolling win-rate (20)"); axs[1, 0].set_ylabel("rolling win-rate (20)")
    axs[0, 0].legend(fontsize=8); fig.suptitle("Chart 1 - Rolling win-rate (mean of 30 seeds); dotted = 33.3% luck")
    fig.tight_layout(); fig.savefig(f"{OUT}/chart1_rolling_winrate.png", dpi=130); plt.close(fig)

    # ---- chart 2: mean win-rate per predictor and bot, error bars (std over seeds)
    fig, ax = plt.subplots(figsize=(12, 5)); w = 0.16
    for i, a in enumerate(AI_NAMES):
        t = tab[tab.p1 == a].set_index("p2").loc[bot_names]
        ax.bar(np.arange(len(bot_names)) + (i - 2) * w, t["mean"], w, yerr=t["std"], capsize=2, label=a)
    ax.axhline(1 / 3, color="k", lw=.8, ls=":"); ax.set_xticks(range(len(bot_names))); ax.set_xticklabels(bot_names)
    ax.set_ylabel("win-rate"); ax.set_title("Chart 2 - Mean win-rate per predictor and bot (error bars = std over 30 seeds)")
    ax.legend(ncol=5, fontsize=8); fig.tight_layout(); fig.savefig(f"{OUT}/chart2_winrate_bars.png", dpi=130); plt.close(fig)

    # ---- chart 3: predictor-vs-predictor heatmap (row player's win-rate)
    H = pd.DataFrame(np.nan, index=AI_NAMES, columns=AI_NAMES)
    pm = pairs.groupby(["p1", "p2"]).win_rate.mean()
    for (a, b), v in pm.items():
        H.loc[a, b] = v; H.loc[b, a] = pairs[(pairs.p1 == a) & (pairs.p2 == b)].loss_rate.mean()
    H.round(3).to_csv(f"{OUT}/tournament_heatmap.csv")
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); im = ax.imshow(H.values.astype(float), cmap="RdYlGn", vmin=.2, vmax=.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(AI_NAMES); ax.set_yticks(range(5)); ax.set_yticklabels(AI_NAMES)
    for i in range(5):
        for j in range(5):
            if i != j: ax.text(j, i, f"{H.values[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("opponent"); ax.set_ylabel("row predictor"); fig.colorbar(im, label="row's win-rate")
    ax.set_title("Chart 3 - Predictor vs predictor"); fig.tight_layout()
    fig.savefig(f"{OUT}/chart3_tournament_heatmap.png", dpi=130); plt.close(fig)

    # ---- chart 4: prediction accuracy and time per move
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.bar(AI_NAMES, metrics.pred_acc); a1.axhline(1 / 3, color="k", ls=":"); a1.set_title("Next-move prediction accuracy (vs bots)")
    a2.bar(AI_NAMES, metrics.ms_per_move, color="C1"); a2.set_title("Average time per move (ms)"); a2.set_yscale("log")
    fig.suptitle("Chart 4"); fig.tight_layout(); fig.savefig(f"{OUT}/chart4_accuracy_time.png", dpi=130); plt.close(fig)
    print("charts saved in", OUT)


if __name__ == "__main__":
    main("--cached" in sys.argv)
