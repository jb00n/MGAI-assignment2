import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------- LOAD DATA ----------

def load_csv(path: str) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "agent": row["Agent"],
                "games": int(row["Games"]),
                "wins": int(row["Wins"]),
                "win_pct": float(row["Win%"]),
                "elo": float(row["ELO"]),
                "ts_mu":float(row["TrueSkill_mu"]),
                "ts_sigma": float(row["TrueSkill_sigma"]),
                "conservative": float(row["Conservative"]),
            })
    # sort by conservative TrueSkill score descending
    rows.sort(key=lambda r: r["conservative"], reverse=True)
    return rows

# ---------- STYLE ----------

COLOURS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

def agent_colours(agents):
    return {a: COLOURS[i % len(COLOURS)] for i, a in enumerate(agents)}

# ---------- bar chart of win% ----------

def plot_winpct(data, out):
    agents = [r["agent"]   for r in data]
    win_pct = [r["win_pct"] for r in data]
    cols = [COLOURS[i % len(COLOURS)] for i in range(len(agents))]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(agents, win_pct, color=cols, width=0.55, zorder=2)
    ax.set_ylabel("Win %")
    ax.set_title("Win Rate by Agent", fontweight="bold")
    ax.set_ylim(0, max(win_pct) * 1.2)
    ax.axhline(25, color="grey", linestyle="--", linewidth=0.8, label="25% (random baseline)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    for bar, val in zip(bars, win_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(out, "1_win_pct.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------- ELO vs TrueSkill scatter ----------

def plot_elo_vs_ts(data, out):
    col_map = agent_colours([r["agent"] for r in data])

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in data:
        ax.scatter(r["elo"], r["conservative"], color=col_map[r["agent"]],
                   s=100, zorder=3)
        ax.annotate(r["agent"], (r["elo"], r["conservative"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.set_xlabel("ELO Rating")
    ax.set_ylabel("TrueSkill Conservative Score (mu - 3·sigma)")
    ax.set_title("ELO vs TrueSkill Conservative Score", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    path = os.path.join(out, "2_elo_vs_trueskill.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------- TrueSkill mu +/- sigma error bars ----------

def plot_trueskill_errorbars(data, out):
    agents = [r["agent"] for r in data]
    mus = [r["ts_mu"] for r in data]
    sigmas = [r["ts_sigma"] for r in data]
    cols = [COLOURS[i % len(COLOURS)] for i in range(len(agents))]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(agents))

    for i, (mu, sigma, col) in enumerate(zip(mus, sigmas, cols)):
        ax.errorbar(mu, i, xerr=sigma, fmt="o", color=col, ecolor=col, elinewidth=2, capsize=6, markersize=8, zorder=3)
        # shade 1-sigma band
        ax.barh(i, 2 * sigma, left=mu - sigma, height=0.35, color=col, alpha=0.15, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agents)
    ax.set_xlabel("TrueSkill mu (± 1 sigma)")
    ax.set_title("TrueSkill Rating with Uncertainty", fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    path = os.path.join(out, "3_trueskill_errorbars.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------- wins vs games played ----------

def plot_wins_vs_games(data, out):
    col_map = agent_colours([r["agent"] for r in data])

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in data:
        losses = r["games"] - r["wins"]
        ax.scatter(r["games"], r["wins"], color=col_map[r["agent"]], s=120, zorder=3, label=r["agent"])
        ax.annotate(r["agent"], (r["games"], r["wins"]),textcoords="offset points", xytext=(6, 4), fontsize=9)

    # diagonal reference lines for win rates
    max_games = max(r["games"] for r in data) * 1.1
    x = np.linspace(0, max_games, 200)
    for rate, style in [(0.25, "--"), (0.50, "-.")]:
        ax.plot(x, rate * x, color="grey", linestyle=style, linewidth=0.8,
                label=f"{int(rate*100)}% win rate")

    ax.set_xlabel("Games Played")
    ax.set_ylabel("Wins")
    ax.set_title("Wins vs Games Played", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(linestyle="--", alpha=0.4, zorder=0)
    plt.tight_layout()
    path = os.path.join(out, "4_wins_vs_games.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------- combined leaderboard (grouped bars) ----------

def plot_combined_leaderboard(data, out):
    agents = [r["agent"] for r in data]
    n = len(agents)
    x = np.arange(n)
    width = 0.25

    # normalise each metric to 0-100 for comparison
    def normalise(vals):
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return [50.0] * len(vals)
        return [100 * (v - lo) / (hi - lo) for v in vals]

    win_pcts = normalise([r["win_pct"]      for r in data])
    elos = normalise([r["elo"]           for r in data])
    conservatives = normalise([r["conservative"] for r in data])

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width, win_pcts, width, label="Win% (normalised)", color="#4C72B0", zorder=2)
    b2 = ax.bar(x, elos, width, label="ELO (normalised)", color="#DD8452", zorder=2)
    b3 = ax.bar(x + width,  conservatives, width, label="TS Conservative (normalised)", color="#55A868", zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=15, ha="right")
    ax.set_ylabel("Normalised Score (0 = worst, 100 = best)")
    ax.set_title("Combined Leaderboard — All Metrics Normalised", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_ylim(0, 115)
    plt.tight_layout()
    path = os.path.join(out, "5_combined_leaderboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

# ---------- MAIN ----------

if __name__ == "__main__":
    
    csv_file = "best_snake_results.csv"
    out_dir  = "plots"
 
    os.makedirs(out_dir, exist_ok=True)
    data = load_csv(csv_file)
 
    print(f"\n  Loaded {len(data)} agents from {csv_file}")
    print(f"  Saving plots to {out_dir}/\n")
 
    plot_winpct(data, out_dir)
    plot_elo_vs_ts(data, out_dir)
    plot_trueskill_errorbars(data, out_dir)
    plot_wins_vs_games(data, out_dir)
    plot_combined_leaderboard(data, out_dir)
 
    print("\n  All done!")