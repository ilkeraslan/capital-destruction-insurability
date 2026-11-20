"""Replication code for the numerical examples, tables and figures.

Running `python replicate.py [output_folder]` prints every number reported in Sections 2, 3.3,
4.3, 4.5, 4.6 and 5, in the notes to the tables and figures and in the online appendix, and writes
fig1_recovery.png and fig2_identification.png to output_folder (default: results).

Stylized economy (all amounts in USD trillion):
  annual world output Y = 100; pool assets K = 1; event threshold a = 0.05 (5% of world output);
  event probability 2% (conditional on survival); surviving assets A(s) = K (1 - s)^theta, theta = 2;
  other claims on the pool C(s) = 8 s (8% of the aggregate loss) in Section 5.2 and Figures 1-2.
Prices are expected payments per dollar promised in units of a bond that pays one unit in
every survival state (Assumption 4), reported in cents.
Requires numpy, scipy and matplotlib.
"""
import os
import sys

import numpy as np
from scipy import integrate, optimize

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

Y, K, A_THRESH, THETA, C_SLOPE, SENIOR_CLASS, P_EVENT = 100.0, 1.0, 0.05, 2.0, 8.0, 0.25, 0.02


def odds(z):
    return np.inf if z >= 1 else z / (1.0 - z)


def inv_odds(o):
    return 1.0 if np.isinf(o) else o / (1.0 + o)


# --------------------------------------------------------------------------- recovery rules
def assets(s):
    return K * (1.0 - s) ** THETA


def other_claims(s):
    return C_SLOPE * s


def r_equal(s, x=0.0):
    """Equal priority (pro rata) on the pool, eq. (1); x -> 0 for a small claim."""
    s = np.asarray(s, float)
    return np.where(s < 1, np.minimum(1.0, assets(s) / (x + other_claims(s))), 0.0)


def r_senior(s, X=SENIOR_CLASS):
    s = np.asarray(s, float)
    return np.where(s < 1, np.minimum(1.0, assets(s) / X), 0.0)


def r_junior(s):
    """Subordinated claim of negligible size: paid in full while A > C, nothing otherwise."""
    s = np.asarray(s, float)
    return np.where((s < 1) & (assets(s) > other_claims(s)), 1.0, 0.0)


def r_intact(s):
    """Equal priority on a pool whose assets stay at K (backing intact), other claims C(s) = 8s."""
    s = np.asarray(s, float)
    return np.where(s < 1, np.minimum(1.0, K / np.maximum(other_claims(s), 1e-12)), 0.0)


def r_coll_falls(s):
    """Own collateral (z = x) invested in an asset whose value falls as (1 - s)^theta."""
    s = np.asarray(s, float)
    return np.where(s < 1, (1.0 - s) ** THETA, 0.0)


S_C = optimize.brentq(lambda s: assets(s) - other_claims(s), 1e-9, 0.99)  # = 5 - sqrt(24)
S1_SENIOR = 1.0 - np.sqrt(SENIOR_CLASS)


def line(label, value):
    print(f"  {label:<62s} {value}")


print("Frontiers")
line("s_c (assets equal other claims), = 5 - sqrt(24)", f"{S_C:.4f}")
line("full-payment frontier of the senior claim", f"{S1_SENIOR:.4f}")
for share in (0.1, 0.001):    # Section 3.3: no other claims, s_1 = 1 - (x/K)^(1/2)
    line(f"no other claims, claim of {share:g} of the assets: s_1", f"{1 - share ** (1 / THETA):.4f}")

# --------------------------------------------------------------------------- Table 3
print("\nTable 3: the same claim under different conditions of payment (severity 0.2)")
s_e = 0.2
A_int, A_er = K, assets(s_e)
C_lo, C_hi = 4.0 * s_e, C_SLOPE * s_e          # other claims of 4% and 8% of the aggregate loss
rows = [
    ("A extinction", None, 0.0),
    ("B intact, C = 0.8, equal", A_int, min(1, A_int / C_lo)),
    ("C eroded, C = 0.8, equal", A_er, min(1, A_er / C_lo)),
    ("D intact, C = 1.6, equal", A_int, min(1, A_int / C_hi)),
    ("E eroded, C = 1.6, equal", A_er, min(1, A_er / C_hi)),
    ("F eroded, C = 1.6, senior class 0.25", A_er, min(1, A_er / SENIOR_CLASS)),
    ("G eroded, C = 1.6, subordinated", A_er, 1.0 if A_er > C_hi else 0.0),
    ("H own collateral, value unaffected", 1.0, 1.0),
    ("I own collateral, value falls as (1-s)^2", A_er, (1 - s_e) ** THETA),
]
for name, backing, r in rows:
    price = 100 * P_EVENT * r
    implied_full = price / 100
    implied_known = "none" if r == 0 else f"{(price / 100) / r:.3%}"
    line(name, f"backing {backing}  paid {r:.4f}  price {price:.3f} c  "
               f"implied(full) {implied_full:.3%}  implied(known) {implied_known}")
r_B, r_C, r_D, r_E = (rows[i][2] for i in (1, 2, 3, 4))
line("claim of USD 1 billion in row E: recovery", f"{A_er / (C_hi + 0.001):.5f}")
line("row E with theta = 1: recovery", f"{K * (1 - s_e) / C_hi:.3f}")
line("price ratio B/C (destruction, other claims 0.8)", f"{r_B / r_C:.3f}")
line("price ratio D/E (destruction, other claims 1.6)", f"{r_D / r_E:.4f}")
line("price ratio B/D (added claims, intact assets)", f"{r_B / r_D:.3f}")
line("price ratio C/E (added claims, eroded assets)", f"{r_C / r_E:.3f}")
line("row C: largest other claims paid in full at s = 0.2", f"{A_er:.2f}")

# --------------------------------------------------------------------------- Table 4
print("\nTable 4: five economies with a price of 1 cent (equal priority, C = 8s)")
q = 0.01
for s in [0.1, 0.2, 0.3, 0.4, 0.6]:
    r = float(r_equal(s))
    p = q / r
    line(f"severity {s:.1f}", f"loss {Y * s:.0f}  backing {assets(s):.2f}  claims {other_claims(s):.2f}  "
                               f"paid {100 * r:.2f} c  p {p:.3%}  expected loss {p * Y * s:.3f}")
line("bounds if severity <= 0.4 (Prop. 3)", f"[{q:.2%}, {q / float(r_equal(0.4)):.3%}]")
line("ratio of bounds = C(0.4)/A(0.4)", f"{other_claims(0.4) / assets(0.4):.3f}")
line("with full payment (p = 1%), expected loss from the event, USD trillion", f"{0.1 * Y * q:.2f} to {0.6 * Y * q:.2f}")

print("\nSensitivity of s_c and of the upper bound (severity <= 0.4) to the slope of other claims")
for slope in (2.0, 4.0, 8.0):
    sc_ = optimize.brentq(lambda s: assets(s) - slope * s, 1e-9, 0.99)
    r04 = min(1.0, assets(0.4) / (slope * 0.4))
    line(f"other claims {slope:.0f}% of the loss: s_c, upper bound", f"{sc_:.3f}, {q / r04:.2%}")
sc_lev = optimize.brentq(lambda s: 4.0 * (1 - s) ** THETA - (3.0 + C_SLOPE * s), 1e-9, 0.99)
line("leverage: assets 4, other claims 3 + 8s: s_c", f"{sc_lev:.3f}")

# --------------------------------------------------------------------------- Table 5 and online Table B1 (Propositions 3 and 6)
print("\nTable 5 (Gamma = 1) and online Table B1: probabilities consistent with a price of 1 cent")


def identified_set(qq, r_a, r_bar, Gamma, kappa_L=1.0):
    """Proposition 6: closed-form bounds of the interval; Gamma = kappa_L = 1 gives Proposition 3."""
    lo = qq / (Gamma * r_a - (Gamma - 1) * qq)
    hi = 1.0 if r_bar <= qq else min(1.0, qq / (kappa_L * r_bar - (kappa_L - 1) * qq))
    return lo, hi


def r_avg(rfun, lo, hi):
    """Average recovery over [lo, hi]: replaces worst-case recovery when the density of
    severity given the event does not rise with severity (Proposition 3(v))."""
    pts = [S_C] if lo < S_C < hi else None
    val, _ = integrate.quad(lambda t: float(rfun(t)), lo, hi, points=pts, limit=500)
    return val / (hi - lo)


def r_lev(s, safe=0.0):
    """Equal priority with leverage: assets 4 (a share `safe` keeps its nominal value), other claims 3 + 8s."""
    s = np.asarray(s, float)
    A = 4.0 * (safe + (1 - safe) * (1 - s) ** THETA)
    return np.where(s < 1, np.minimum(1.0, A / (3.0 + C_SLOPE * s)), 0.0)


claims = [("paid in full at every severity", 1.0, 1.0),
          ("equal priority, severity <= 0.4", 1.0, float(r_equal(0.4))),
          ("equal priority, severity <= 0.6", 1.0, float(r_equal(0.6))),
          ("equal priority, backing intact, severity <= 0.6", 1.0, float(r_intact(0.6))),
          ("equal priority, severity <= 0.6, density not rising", 1.0, r_avg(r_equal, A_THRESH, 0.6)),
          ("equal priority, severity <= 0.6, leverage", float(r_lev(A_THRESH)), float(r_lev(0.6))),
          ("equal priority, severity <= 0.6, leverage, half safe", float(r_lev(A_THRESH, 0.5)), float(r_lev(0.6, 0.5))),
          ("subordination, severity <= 0.4", 1.0, 0.0)]
COLS = [(1, 1.0), (40, 1.0), (40, 0.5)]   # (Gamma, kappa_L); Gamma = 8 gives a lower bound of 0.126%
line("lower bound with Gamma = 8 (claim paid in full at the threshold)", f"{identified_set(q, 1.0, 1.0, 8)[0]:.3%}")
for name, ra, rb in claims:
    print(f"  {name}: recovery at the worst severity (or average) {100 * rb:.2f} cents")
    for (G, kL) in COLS + [(8, 1.0)]:
        lo, hi = identified_set(q, ra, rb, G, kL)
        line(f"    Gamma = {G}, kappa_L = {kL}", f"[{lo:.3%}, {hi:.3%}]")
line("density not rising, severity <= 0.99: upper bound", f"{q / r_avg(r_equal, A_THRESH, 0.99):.2%}")
# Section 5.2: the levered pool if none of its assets erodes (assets 4, other claims 3 + 8s), severity <= 0.6
line("leverage, backing intact, severity <= 0.6: upper bound", f"{q / min(1.0, 4.0 / (3.0 + C_SLOPE * 0.6)):.2%}")
# Section 5.2, Proposition 1(v): half the assets safe, so phi >= 1/2; other claims at most 3 + 8 = 11
line("leverage, half safe: floor on recovery 2/11, limit of the upper bound", f"{4.0 * 0.5 / 11.0:.4f}, {q / (4.0 * 0.5 / 11.0):.2%}")
# Section 4.3: for a claim with equal priority paid in full at the threshold, the ratio of the bounds of the
# interval is (x + C(s_bar))/K, over-commitment, times 1/phi(s_bar), capital destruction (x -> 0 here)
print("\nRatio of the bounds of the interval: claims-to-assets ratio (x + C)/K times capital destruction 1/phi")
for name, K_, C_, phi_ in [("row C pool (other claims 4s), severity <= 0.2", 1.0, 4.0 * 0.2, (1 - 0.2) ** THETA),
                           ("unlevered pool, severity <= 0.4", 1.0, C_SLOPE * 0.4, (1 - 0.4) ** THETA),
                           ("unlevered pool, severity <= 0.6", 1.0, C_SLOPE * 0.6, (1 - 0.6) ** THETA),
                           ("backing intact, severity <= 0.6", 1.0, C_SLOPE * 0.6, 1.0),
                           ("leverage, severity <= 0.6", 4.0, 3.0 + C_SLOPE * 0.6, (1 - 0.6) ** THETA),
                           ("leverage, half safe, severity <= 0.6", 4.0, 3.0 + C_SLOPE * 0.6, 0.5 + 0.5 * (1 - 0.6) ** THETA)]:
    over, destr = C_ / K_, 1.0 / phi_
    line(f"  {name}", f"{over:.2f} x {destr:.2f} = {over * destr:.2f}")

# Online Appendix B.6 (Proposition B5, Remark B1): density condition on the true distribution, kernel rising with severity
S_UP = 0.99
s_star = optimize.brentq(lambda s: float(r_equal(s)) - q, S_C, S_UP)
for G_ in (8, 40):
    lo_part, _ = integrate.quad(lambda s: float(r_equal(s)) - q, A_THRESH, s_star, points=[S_C], limit=500)
    hi_part, _ = integrate.quad(lambda s: float(r_equal(s)) - q, s_star, S_UP, limit=500)
    term = (lo_part + G_ * hi_part) / (S_UP - A_THRESH)
    line(f"B.6, Remark B1: uniform severity on [a, 0.99], step kernel 1 -> {G_}: p", f"{q / (q + term):.3%}")
line("density not rising, subordination, severity <= 0.4: upper bound",
     f"{q / r_avg(r_junior, A_THRESH, 0.4):.2%}")
line("CRRA 4 kernel ratio at severity 0.4 and 0.6", f"{0.6 ** -4:.2f}, {0.4 ** -4:.2f}")
# Figure 2(a): where the upper bounds reach 100 percent, and their limits
line("Fig. 2(a): equal priority, upper bound reaches 100% at s_bar", f"{s_star:.4f}")
line("Fig. 2(a): falling collateral, upper bound reaches 100% at s_bar", f"{1 - np.sqrt(q):.4f}")
line("Fig. 2(a): backing intact, limit of the upper bound as s_bar -> 1", f"{q / (K / C_SLOPE):.2%}")
line("Fig. 2(a): falling collateral, recovery and both bounds at the threshold",
     f"{float(r_coll_falls(A_THRESH)):.4f}, {q / float(r_coll_falls(A_THRESH)):.3%}")

print("\nKnown CRRA kernel m ~ (1-s)^-gamma, two-point economies (non-event severity 0)")
for gamma in (2, 4):
    ps = []
    for s in [0.1, 0.2, 0.3, 0.4, 0.6]:
        r = float(r_equal(s))
        ps.append(inv_odds(odds(q / r) * (1 - s) ** gamma))
    line(f"gamma = {gamma}", "  ".join(f"{p:.2%}" for p in ps))

# --------------------------------------------------------------------------- Section 5.3: ladder
print("\nLadder economy: truncated Pareto severity, index 1.5, on [a, 0.8], event probability 2%")
ALPHA, S_BAR = 1.5, 0.8
norm = A_THRESH ** (-ALPHA) - S_BAR ** (-ALPHA)


def G_true(u):
    return P_EVENT * (u ** (-ALPHA) - S_BAR ** (-ALPHA)) / norm


def g_true(u):
    return P_EVENT * ALPHA * u ** (-ALPHA - 1) / norm


def q_ladder(u, rfun=r_equal):
    pts = [S_C] if u < S_C else None
    val, _ = integrate.quad(lambda t: float(rfun(t)) * g_true(t), u, S_BAR, points=pts, limit=500)
    return val


for u in [0.05, 0.1, 0.2, 0.4, 0.6]:
    Gu, qu = G_true(u), q_ladder(u)
    line(f"u = {u:.2f}", f"G {Gu:.5%}  price {100 * qu:.5f} c  ratio {qu / Gu:.4f}")


def eps_G(u):
    return u * g_true(u) / G_true(u)


def eps_q(u):
    return u * float(r_equal(u)) * g_true(u) / q_ladder(u)


for u in [0.1, 0.2, 0.4]:
    line(f"local tail index at u = {u}", f"true {eps_G(u):.3f}  from prices {eps_q(u):.3f}")
for lo_, hi_ in [(0.1, 0.2), (0.2, 0.4), (0.4, 0.6)]:
    sG = -np.log(G_true(hi_) / G_true(lo_)) / np.log(hi_ / lo_)
    sq = -np.log(q_ladder(hi_) / q_ladder(lo_)) / np.log(hi_ / lo_)
    line(f"log-log slope on [{lo_}, {hi_}]", f"true {sG:.3f}  from prices {sq:.3f}")
layer_true, _ = integrate.quad(G_true, A_THRESH, S_BAR, limit=500)
layer_price, _ = integrate.quad(q_ladder, A_THRESH, S_BAR, limit=500)
line("expected loss in layer above a: prices / true", f"{layer_price / layer_true:.4f}")
line("recovery at 0.6 and 0.8 (error amplification 1/r)", f"{float(r_equal(0.6)):.4f}, {float(r_equal(0.8)):.5f}"
     f"  (1/r = {1 / float(r_equal(0.6)):.0f} and {1 / float(r_equal(0.8)):.0f})")
rbar_l = r_avg(r_equal, A_THRESH, S_BAR)
line("one claim at 5%: bounds, severity <= 0.8", f"[{q_ladder(A_THRESH) / float(r_equal(A_THRESH)):.3%}, 100%]")
line("same, density not rising: average recovery, upper bound", f"{rbar_l:.4f}, {q_ladder(A_THRESH) / rbar_l:.2%}")


def kernel_ladder(gamma):
    """Ladder prices under a kernel proportional to (1-s)^-gamma, no loss outside the event."""
    Em_event, _ = integrate.quad(lambda s: (1 - s) ** (-gamma) * g_true(s) / P_EVENT, A_THRESH, S_BAR, limit=500)
    Em = (1 - P_EVENT) + P_EVENT * Em_event

    def qk(u):
        pts = [S_C] if u < S_C else None
        val, _ = integrate.quad(lambda t: (1 - t) ** (-gamma) / Em * float(r_equal(t)) * g_true(t),
                                u, S_BAR, points=pts, limit=500)
        return val
    slope = -np.log(qk(0.4) / qk(0.2)) / np.log(2.0)
    lay, _ = integrate.quad(qk, A_THRESH, S_BAR, limit=500)
    return slope, lay / layer_true


for gamma_ in (2, 4):
    sl, lr = kernel_ladder(gamma_)
    line(f"kernel gamma = {gamma_}: log-log slope [0.2, 0.4], layer ratio", f"{sl:.3f}, {lr:.3f}")

# kappa below one (Section 4.6): AI raises output by half with probability 10%; the event (loss 20%)
# occurs with probability 2% and only in those worlds; relative risk aversion 4
probs = np.array([0.90, 0.08, 0.02])
cons = np.array([1.0, 1.5, 1.5 * 0.8])
mk = cons ** -4.0
mk = mk / (probs * mk).sum()
kappa_ex = mk[2] / ((probs[0] * mk[0] + probs[1] * mk[1]) / (probs[0] + probs[1]))
line("kappa in the AI-growth example; price of full claim (cents)", f"{kappa_ex:.3f}, {100 * probs[2] * mk[2]:.3f}")


def ladder_bounds(J):
    us = list(A_THRESH * (S_BAR / A_THRESH) ** (np.arange(J) / J)) + [S_BAR]
    qs = [q_ladder(u) for u in us[:-1]] + [0.0]
    lo = sum((qs[j] - qs[j + 1]) / float(r_equal(us[j])) for j in range(J))
    r_end = [float(r_equal(us[j + 1])) for j in range(J)]
    hi = sum((qs[j] - qs[j + 1]) / r_end[j] for j in range(J))
    return lo, min(1.0, hi)


for J in (1, 5, 10, 30):
    lo, hi = ladder_bounds(J)
    line(f"finite ladder, {J} thresholds (log-spaced)", f"[{lo:.3%}, {hi:.3%}]")
line("five thresholds (percent of world output)",
     ", ".join(f"{100 * u:.1f}" for u in A_THRESH * (S_BAR / A_THRESH) ** (np.arange(5) / 5)))

# --------------------------------------------------------------------------- Section 4.2 and 4.4
print("\nRiskless rate with survivable disasters (Appendix B.5)")
gamma, b = 4.0, 0.2
for R_B in (1.0, 0.8 ** 4):
    J = (1 - b) ** (-gamma) * R_B - 1
    first_order = -0.01 * J * 1e4
    exact = -(np.log(1 + 0.02 * J) - np.log(1 + 0.01 * J)) * 1e4
    line(f"p_d from 1% to 2%, bond recovery {R_B:.3f}", f"first order {first_order:.1f} bp, exact (log) {exact:.1f} bp")
print("\nPrecision of the complement route (Appendix B.1)")
p, D_err = P_EVENT, 0.005
line("complement route, D understated/overstated by 0.5%",
     f"{(1 - (1 - p) / (1 - D_err)) / p - 1:+.2%} / {(1 - (1 - p) / (1 + D_err)) / p - 1:+.2%}")
line("event-claim route, same errors", f"{1 / (1 - D_err) - 1:+.2%} / {1 / (1 + D_err) - 1:+.2%}")
line("ten-year horizon: 5 bp a year changes D by", f"{1 - np.exp(-0.0005 * 10):.3%}")
line("ratio of error magnitudes, (1 - p)/p", f"{(1 - p) / p:.0f}")
line("loss of 5% of world output relative to USD 715bn", f"{0.05 * Y / 0.715:.2f}")
line("hazards implied by 0.38% and 3% by 2100 (77 years), bp a year",
     f"{-np.log(1 - 0.0038) / 77 * 1e4:.2f}, {-np.log(1 - 0.03) / 77 * 1e4:.2f}")
line("collateral needed for full payment at severity 0.4 (z/x)", f"{1 / (1 - 0.4) ** 2:.2f}")

# --------------------------------------------------------------------------- figures
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    for f in ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"]:
        try:
            font_manager.fontManager.addfont(f)
        except Exception:
            pass
    plt.rcParams.update({"font.family": "Liberation Serif", "mathtext.fontset": "custom",
                         "mathtext.rm": "Liberation Serif", "mathtext.it": "Liberation Serif:italic",
                         "font.size": 10, "axes.linewidth": 0.8, "legend.frameon": False, "savefig.dpi": 300})

    # Figure 1: recovery by rule
    s = np.linspace(A_THRESH, 0.995, 4000)
    fig, ax = plt.subplots(figsize=(6.3, 4.0))
    ax.axvspan(0, A_THRESH, color="0.92", lw=0)
    ax.text(A_THRESH / 2, 0.5, "no event", rotation=90, ha="center", va="center", fontsize=8.5, color="0.35")
    ax.plot(s, np.ones_like(s), color="0.55", lw=2.2, label="Separate collateral, value unaffected by the loss")
    ax.plot(s, r_senior(s), color="black", lw=1.3, ls=(0, (5, 2)), label="Seniority (senior class USD 0.25 trillion)")
    ax.plot(s, r_coll_falls(s), color="black", lw=1.1, ls=(0, (1, 1.6)), label="Separate collateral, value falls with the loss")
    ax.plot(s, r_equal(s), color="black", lw=1.6, label="Equal priority")
    sj = np.linspace(A_THRESH, 0.995, 20000)
    ax.plot(sj, r_junior(sj), color="black", lw=1.1, ls=(0, (6, 2, 1, 2)), label="Subordination")
    ax.plot([1.0], [1.0], marker="o", mfc="white", mec="0.45", mew=1.0, ms=4.5, clip_on=False, zorder=5)
    ax.plot([1.0], [0.0], marker="o", color="black", ms=4.5, clip_on=False)
    ax.annotate("extinction:\nnothing paid", xy=(1.0, 0.0), xytext=(0.87, 0.2), fontsize=8.5, color="0.25",
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    for xv, lab in [(S_C, r"$s_c$"), (S1_SENIOR, r"$s_1$ (seniority)")]:
        ax.axvline(xv, color="0.4", lw=0.7, ls=":")
        ax.text(xv + 0.008, 1.035, lab, fontsize=10, color="0.25")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.035, 1.1)
    ax.set_xlabel("Severity $s$ (share of world output lost)")
    ax.set_ylabel("Recovery rate $r(s)$")
    ax.set_xticks([0, A_THRESH, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "$a$", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    hs, ls_ = ax.get_legend_handles_labels()
    order = [ls_.index(k) for k in ["Equal priority", "Seniority (senior class USD 0.25 trillion)", "Subordination",
                                     "Separate collateral, value unaffected by the loss",
                                     "Separate collateral, value falls with the loss"]]
    ax.legend([hs[i] for i in order], [ls_[i] for i in order], loc="upper center", fontsize=8.4,
              bbox_to_anchor=(0.5, -0.17), ncol=2, handlelength=3.2, columnspacing=1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_recovery.png"))
    plt.close(fig)

    # Figure 2: (a) upper bounds implied by a price of 1 cent; (b) ladder
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.45))
    sv = np.linspace(A_THRESH, 0.99, 1000)
    rbar_v = np.array([r_avg(r_equal, A_THRESH, t) if t > A_THRESH + 1e-9 else 1.0 for t in sv])
    ax1.plot(sv, np.full_like(sv, q), color="0.55", lw=2.2, label="Paid in full")
    ax1.plot(sv, np.minimum(q / r_intact(sv), 1), color="black", lw=1.1, ls=(0, (6, 2, 1, 2)),
             label="Backing intact")
    ax1.plot(sv, np.minimum(q / rbar_v, 1), color="black", lw=1.1, ls=(0, (4, 2)),
             label="Density not rising")
    ax1.plot(sv, np.minimum(q / r_coll_falls(sv), 1), color="black", lw=1.1, ls=(0, (1, 1.6)),
             label="Collateral falls")
    ax1.plot(sv, np.minimum(q / r_equal(sv), 1), color="black", lw=1.6, label="Equal priority")
    ax1.axhline(q / (K / C_SLOPE), color="0.6", lw=0.6, ls=":")
    ax1.text(0.015, 0.08 * 1.08, "8%: limit with\nbacking intact", fontsize=7.8, color="0.3", ha="left", va="bottom")
    for xv in (s_star, 1 - np.sqrt(q)):        # where the upper bounds reach 100 percent
        ax1.plot([xv], [1.0], marker="o", color="black", ms=3.6, clip_on=False, zorder=5)
    for xv in (s_star, 1 - np.sqrt(q)):
        ax1.text(xv + 0.014, 0.80, f"{xv:.2f}", fontsize=7.8, color="0.25", ha="left", va="center")
    ax1.set_yscale("log")
    ax1.set_ylim(0.007, 1.0)
    ax1.set_xlim(0, 1.0)
    ax1.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0])
    ax1.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax1.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax1.set_xlabel(r"Worst severity the observer considers possible, $\overline{s}$")
    ax1.set_ylabel("Upper bound on the event probability")
    ax1.set_title("(a) Upper bounds implied by a price of 1 cent", fontsize=9.6)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    hs, ls_ = ax1.get_legend_handles_labels()
    order = [ls_.index(k) for k in ["Paid in full", "Equal priority", "Density not rising", "Backing intact",
                                     "Collateral falls"]]
    ax1.legend([hs[i] for i in order], [ls_[i] for i in order], loc="upper left", fontsize=7.8,
               handlelength=2.4, borderaxespad=0.2, labelspacing=0.25)

    u = np.linspace(A_THRESH, S_BAR - 1e-6, 160)
    ax2.plot(u, 100 * G_true(u), color="0.55", lw=2.2, label="Probability $G(u)$, %")
    ax2.plot(u, [100 * q_ladder(t) for t in u], color="black", lw=1.6, label="Price, cents per dollar")
    for ut in A_THRESH * (S_BAR / A_THRESH) ** (np.arange(5) / 5):    # the five thresholds of Section 5.3
        ax2.plot([ut, ut], [1e-5, 1.8e-5], color="black", lw=0.8)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-5, 5)
    ax2.set_xlim(0, 0.8)
    ax2.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax2.set_xticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
    ax2.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax2.set_yticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    ax2.set_xlabel("Attachment point $u$ (share of world output)")
    ax2.set_ylabel("Probability (%) or price (cents per dollar)")
    ax2.set_title("(b) A ladder of claims", fontsize=9.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", fontsize=7.8, handlelength=2.4)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(os.path.join(OUT_DIR, "fig2_identification.png"))
    plt.close(fig)
    print(f"\nwrote fig1_recovery.png and fig2_identification.png to {os.path.relpath(OUT_DIR)}")
