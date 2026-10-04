from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from portfolio_qaoa import bitstring, int_to_bits, portfolio_stats

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"


def _style(ax, fig):
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def fig_frontier(r: dict):
    q, n = r["qubo"], r["qubo"].n
    feas = np.where(q.feasible_mask())[0]
    pts = np.array([[portfolio_stats(int_to_bits(b, n), q.mu, q.sigma)[m] for m in ("risk", "return")]
                    for b in feas])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style(ax, fig)
    ax.scatter(pts[:, 0] * 100, pts[:, 1] * 100, s=22, color=MUTED, label="All valid portfolios", zorder=2)

    def mark(b, color, label, marker, size, dy):
        s = portfolio_stats(int_to_bits(b, n), q.mu, q.sigma)
        ax.scatter([s["risk"] * 100], [s["return"] * 100], s=size, color=color, marker=marker,
                   edgecolor=SURFACE, linewidth=2, zorder=4, label=label)
        return s

    items = [("optimal", r["optimal"], ORANGE, "Optimal (brute force)", "o", 300),
             ("greedy", r["greedy"], AQUA, "Greedy baseline", "s", 100)]
    if r["qaoa_best"] is not None:
        items.append(("qaoa", r["qaoa_best"], BLUE, "QAOA best sample", "D", 55))
    for _, b, c, lab, m, sz in items:
        mark(b, c, lab, m, sz, 0)
    ax.set_xlabel("Portfolio risk (volatility, %)")
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Where the solvers land among all valid portfolios", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower right")
    fig.tight_layout()
    return fig


def fig_distribution(r: dict, top: int = 12):
    probs, n = r["result"].probs, r["qubo"].n
    idx = np.argsort(probs)[::-1][:top]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _style(ax, fig)
    ax.grid(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    colors = [ORANGE if b == r["optimal"] else BLUE for b in idx]
    ax.bar(range(len(idx)), probs[idx] * 100, color=colors, width=0.62)
    ax.axhline(r["p_random_optimal"] * 100, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(idx) - 0.6, r["p_random_optimal"] * 100, " random guess", va="bottom", ha="right",
            fontsize=8, color=INK2)
    ax.set_xticks(range(len(idx)))
    ax.set_xticklabels([bitstring(int(b), n) for b in idx], rotation=45, ha="right", fontsize=8,
                       family="monospace")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Final QAOA state, top 12 portfolios (orange = true optimum)", loc="left", fontsize=11)
    ax.text(0.99, 0.93, "The answer is the lowest-energy portfolio among the sampled shots,\n"
            "not necessarily the single most likely bitstring.", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    return fig


def fig_convergence(r: dict):
    from matplotlib.ticker import MaxNLocator

    h = np.minimum.accumulate(np.asarray(r["result"].history, dtype=float))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    _style(ax, fig)
    ax.plot(range(len(h)), h, color=BLUE, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Circuit evaluations at the final depth")
    ax.set_ylabel("Best CVaR so far")
    ax.set_title("Classical optimiser tuning the circuit angles", loc="left", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import sys

    from portfolio_qaoa import solve

    out = sys.argv[1] if len(sys.argv) > 1 else "."
    res = solve(backend="numpy")
    for name, fn in (("frontier", fig_frontier), ("distribution", fig_distribution),
                     ("convergence", fig_convergence)):
        fn(res).savefig(f"{out}/{name}.png", dpi=130)
        print("wrote", f"{out}/{name}.png")