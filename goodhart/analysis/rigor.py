"""Statistically honest verdicts for the Goodhart-on-the-Bridge predictions.

This module is the rigorous counterpart to ``goodhart.analysis.curve``. Where
``curve.evaluate_predictions`` reports a bare 'supported'/'refuted' from the sign
of a difference, this module refuses to call anything 'supported' unless the
effect clears a significance/power bar. It REUSES the operator's existing
statistics toolkit ``ztare.experiment_stats`` (it never reimplements the stats);
the only thing here is the wiring from ``RunResult`` fields to those primitives.

Toolkit functions used (read-only dependency)::

    bootstrap_ci(values, statistic=None, n_boot=2000, ci=0.95, seed=42)
        -> (point, lo, hi)
    paired_permutation_test(a, b, n_perm=5000, seed=42, ci_level=0.95)
        -> {"n_paired","observed_delta","ci_lo","ci_hi","p_value", ...}
    spearman_rho_with_ci(xs, ys, ci=0.95)        -> (rho, lo, hi)
    power_aware_verdict(rho, n, target_rho=0.30, alpha=0.05, power=0.80)
        -> (verdict, note)  # verdict in {h1_supported,h0_kept,
                            #             inconclusive_underpowered,invalid_run}
    n_required_for_rho(rho, alpha=0.05, power=0.80, spearman=True) -> N
    detectable_rho_at_n(n, alpha=0.05, power=0.80, spearman=True)  -> |rho|

Graceful degradation: if the toolkit repo is absent, the import fails, and every
public function returns a result dict whose ``verdict`` is ``"stats_unavailable"``
rather than crashing. ``stats_available()`` exposes the state for callers/tests.
"""

from __future__ import annotations

from collections import defaultdict

# -- Make the operator's toolkit importable, then import it ----------------
# bootstrap_paths() adds ~/figs_activist_loop/src (ztare) and
# ~/cognitive-firm/src to sys.path. If the repo is absent the import below
# fails and we degrade to the stats-unavailable path instead of crashing.
try:
    from ..llm.base import bootstrap_paths

    bootstrap_paths()
except Exception:  # pragma: no cover - base is always importable in-tree
    pass

try:
    from ztare.experiment_stats import (  # type: ignore
        bootstrap_ci,
        paired_permutation_test,
        spearman_rho_with_ci,
        power_aware_verdict,
        n_required_for_rho,
        detectable_rho_at_n,
        bh_fdr,
    )

    _STATS_OK = True
except Exception:  # pragma: no cover - exercised via the unavailable test
    _STATS_OK = False
    bootstrap_ci = paired_permutation_test = spearman_rho_with_ci = None  # type: ignore
    power_aware_verdict = n_required_for_rho = detectable_rho_at_n = None  # type: ignore
    bh_fdr = None  # type: ignore

# Minimum paired observations the toolkit's permutation test will evaluate.
_MIN_PAIRED = 5
# Target effect size for the P3 power analysis (the lead's pre-registered rho).
_TARGET_RHO = 0.30
_UNAVAILABLE = "stats_unavailable"


def stats_available() -> bool:
    """Whether ``ztare.experiment_stats`` imported. Tests monkeypatch this path."""
    return _STATS_OK


def _unavailable(**extra) -> dict:
    """Uniform degraded result: never crash, never claim a verdict we can't back."""
    base = {"verdict": _UNAVAILABLE,
            "note": "ztare.experiment_stats not importable; install the toolkit repo."}
    base.update(extra)
    return base


# -- RunResult plumbing (read-only; mirrors curve.py conventions) ----------
def _no_feedback(result) -> int:
    return int(result.config.get("no_feedback_epochs", 2))


def _post_feedback_gaps(result) -> list:
    """Gap sequence after the opening no-feedback phase (pressure-on window)."""
    k = _no_feedback(result)
    return list(result.gaps[k:]) if len(result.gaps) > k else list(result.gaps)


def _metric(result, metric: str) -> float:
    """The per-run scalar for the curve: terminal gap or post-feedback mean gap."""
    if metric == "mean":
        post = _post_feedback_gaps(result)
        return sum(post) / len(post) if post else 0.0
    return float(result.terminal_gap)


def _by_arch(results: list) -> dict:
    groups: dict = defaultdict(list)
    for r in results:
        groups[r.config["architecture"]].append(r)
    return groups


def _by_lam(results: list) -> dict:
    groups: dict = defaultdict(list)
    for r in results:
        groups[float(r.config["lam"])].append(r)
    return groups


# ======================================================================
# 1. Curve with bootstrap CIs per lambda
# ======================================================================
def curve_with_ci(results: list, arch: str | None = None,
                  metric: str = "terminal") -> dict:
    """Per-lambda point estimate + bootstrap 95% CI of the run metric.

    Groups runs by lambda (optionally filtering to one architecture), collects
    the per-seed metric in each cell, and bootstraps a mean CI via the toolkit's
    ``bootstrap_ci``.

    Args:
        results: list of ``RunResult``.
        arch: optional architecture code ('A'/'B'/'C') to filter to one curve.
        metric: ``"terminal"`` (terminal_gap) or ``"mean"`` (post-no-feedback
            mean gap).

    Returns:
        ``{lam: {"point","ci_lo","ci_hi","n","values"}}`` sorted by lambda, or a
        ``stats_unavailable`` dict if the toolkit is absent.
    """
    if not _STATS_OK:
        return _unavailable(curve={})
    pool = [r for r in results if arch is None or r.config["architecture"] == arch]
    out: dict = {}
    for lam in sorted(_by_lam(pool).keys()):
        cell = _by_lam(pool)[lam]
        values = [_metric(r, metric) for r in cell]
        point, lo, hi = bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42)
        out[lam] = {"point": point, "ci_lo": lo, "ci_hi": hi,
                    "n": len(values), "values": values}
    return out


# ======================================================================
# 2. P2 - Goodhart curve: interior minimum AND a STATISTICALLY REAL dip
# ======================================================================
def p2_goodhart_verdict(results: list, arch: str = "A") -> dict:
    """Is the lambda->G curve non-monotone with a *significant* interior dip?

    Two things must hold to call P2 supported:
      1. the seed-mean terminal-gap curve has an interior minimum, AND
      2. that interior cell sits *significantly* below both shoulders -
         a paired permutation test (interior-min vs lambda=0.8, and vs
         lambda=0) clears p<0.05 with enough paired observations.

    The permutation test pairs runs by seed; with too few seeds the toolkit
    returns ``observed_delta=None`` and we report ``inconclusive_underpowered``
    instead of 'supported'. Per-cell bootstrap CIs are included.

    Returns:
        ``{"arch","min_lam","interior_dip_p","right_arm_p","cells","verdict",
        "note"}``.
    """
    if not _STATS_OK:
        return _unavailable(arch=arch, min_lam=None,
                            interior_dip_p=None, right_arm_p=None)

    arch_runs = [r for r in results if r.config["architecture"] == arch]
    cells_ci = curve_with_ci(arch_runs, arch=arch, metric="terminal")
    lams = sorted(cells_ci.keys())
    if len(lams) < 3:
        return {"arch": arch, "min_lam": None, "interior_dip_p": None,
                "right_arm_p": None, "cells": cells_ci, "verdict": "inconclusive",
                "note": f"need >=3 lambda values on arch {arch}; have {len(lams)}."}

    means = [cells_ci[l]["point"] for l in lams]
    lo_idx = min(range(len(means)), key=lambda i: means[i])
    min_lam = lams[lo_idx]
    interior = 0 < lo_idx < len(means) - 1

    by_lam = _by_lam(arch_runs)

    def _paired_p(a_lam: float, b_lam: float):
        """Permutation-test the interior cell against a shoulder, paired by seed.

        Returns (p_value, observed_delta, ci_lo, ci_hi, n_paired). p_value/delta
        are None when the toolkit declines for too few paired observations.
        """
        a_by_seed = {r.config["seed"]: float(r.terminal_gap) for r in by_lam.get(a_lam, [])}
        b_by_seed = {r.config["seed"]: float(r.terminal_gap) for r in by_lam.get(b_lam, [])}
        seeds = sorted(set(a_by_seed) & set(b_by_seed))
        a = [a_by_seed[s] for s in seeds]
        b = [b_by_seed[s] for s in seeds]
        res = paired_permutation_test(a, b, n_perm=5000, seed=42)
        return (res.get("p_value"), res.get("observed_delta"),
                res.get("ci_lo"), res.get("ci_hi"), res.get("n_paired", len(seeds)))

    right_p = right_delta = left_p = left_delta = None
    n_paired = 0
    if interior:
        right_p, right_delta, r_lo, r_hi, n_paired = _paired_p(min_lam, 0.8)
        left_p, left_delta, _, _, _ = _paired_p(min_lam, 0.0)

    # Verdict logic - honest about both shape and significance/power.
    if not interior:
        verdict = "refuted" if (means and means[0] <= min(means)) else "inconclusive"
        note = (f"minimum at endpoint lambda={min_lam} (not an interior Goodhart "
                f"dip); shape is monotone-ish.")
    elif right_p is None or left_p is None:
        verdict = "inconclusive_underpowered"
        note = (f"interior min at lambda={min_lam} but only n_paired={n_paired} "
                f"(<{_MIN_PAIRED}) seeds -> permutation test cannot resolve; "
                f"need more seeds.")
    elif right_p < 0.05 and left_p < 0.05:
        # The dip is below BOTH shoulders AND both gaps are negative (a real dip,
        # not a spurious rise mislabelled). observed_delta = interior - shoulder.
        real_dip = (right_delta is not None and right_delta < 0
                    and left_delta is not None and left_delta < 0)
        if real_dip:
            verdict = "h1_supported"
            note = (f"interior min at lambda={min_lam} sits significantly below both "
                    f"shoulders (vs lambda=0 p={left_p:.4f}, vs lambda=0.8 "
                    f"p={right_p:.4f}).")
        else:
            verdict = "inconclusive"
            note = (f"interior min at lambda={min_lam} significant but deltas not both "
                    f"negative (left d={left_delta}, right d={right_delta}).")
    else:
        verdict = "h0_kept"
        note = (f"interior min at lambda={min_lam} but dip not significant "
                f"(vs lambda=0 p={left_p}, vs lambda=0.8 p={right_p}); "
                f"cannot reject a flat/monotone curve.")

    return {"arch": arch, "min_lam": min_lam,
            "interior_dip_p": left_p, "right_arm_p": right_p,
            "left_arm_delta": left_delta, "right_arm_delta": right_delta,
            "n_paired": n_paired, "cells": cells_ci,
            "verdict": verdict, "note": note}


# ======================================================================
# 3. P3 - Gaming scales with pressure: Spearman rho + power-aware verdict
# ======================================================================
def p3_gaming_verdict(results: list) -> dict:
    """Does gaming-event count rise with lambda? Power-aware Spearman verdict.

    Builds xs = lambda per run, ys = gaming_event_count per run (every run, not
    cell-averaged - that is the honest N for the correlation). Computes
    ``spearman_rho_with_ci`` and resolves with ``power_aware_verdict`` against
    ``target_rho=0.30``. Also reports ``detectable_rho_at_n`` so an
    underpowered design is visible. Adds a permutation null on the gaming counts
    (paired low-lambda vs high-lambda half) when feasible.

    This NEVER reports 'supported' when underpowered: the verdict is whatever
    ``power_aware_verdict`` returns (h1_supported / h0_kept /
    inconclusive_underpowered / invalid_run).

    Returns:
        ``{"rho","ci","n","detectable_rho_at_n","perm_p","verdict","note"}``.
    """
    if not _STATS_OK:
        return _unavailable(rho=None, ci=(None, None), n=0,
                            detectable_rho_at_n=None, perm_p=None)

    xs = [float(r.config["lam"]) for r in results]
    ys = [float(r.gaming_event_count) for r in results]
    n = len(xs)

    if n < 4:
        return {"rho": None, "ci": (None, None), "n": n,
                "detectable_rho_at_n": detectable_rho_at_n(n),
                "perm_p": None, "verdict": "invalid_run",
                "note": f"n={n} < 4: Spearman undefined."}

    rho, lo, hi = spearman_rho_with_ci(xs, ys, ci=0.95)
    verdict, note = power_aware_verdict(rho if rho is not None else 0.0, n,
                                        target_rho=_TARGET_RHO)
    det = detectable_rho_at_n(n)

    # Permutation null: split runs at the lambda median, permutation-test the two
    # halves' gaming counts (paired by rank index when sizes match) - a direct,
    # distribution-free check that high-lambda games more.
    perm_p = None
    order = sorted(range(n), key=lambda i: xs[i])
    half = n // 2
    low_idx, high_idx = order[:half], order[-half:]
    if half >= _MIN_PAIRED:
        low = [ys[i] for i in low_idx]
        high = [ys[i] for i in high_idx]
        perm_p = paired_permutation_test(high, low, n_perm=5000, seed=42).get("p_value")

    return {"rho": rho, "ci": (lo, hi), "n": n,
            "detectable_rho_at_n": det, "perm_p": perm_p,
            "verdict": verdict, "note": note}


# ======================================================================
# 4. Power planning: how many seeds do we actually need?
# ======================================================================
def power_planning(target_rho: float = _TARGET_RHO) -> dict:
    """Seeds/runs needed to detect ``target_rho`` at alpha=0.05, power=0.80.

    Returns:
        ``{"target_rho","n_required","alpha","power"}`` (or stats-unavailable).
    """
    if not _STATS_OK:
        return _unavailable(target_rho=target_rho, n_required=None)
    return {"target_rho": target_rho,
            "n_required": n_required_for_rho(target_rho, alpha=0.05, power=0.80),
            "alpha": 0.05, "power": 0.80}


# ======================================================================
# 5. Honest scorecard: P2 + P3 (+ P5 descriptor) with one-line summaries
# ======================================================================
def _summary_line(pid: str, verdict: str, detail: str) -> str:
    """One human-readable line; honest about underpowered/unavailable states."""
    label = {
        "h1_supported": "SUPPORTED",
        "h0_kept": "null kept (no effect at target size)",
        "inconclusive_underpowered": "UNDERPOWERED (cannot conclude)",
        "inconclusive": "inconclusive",
        "invalid_run": "invalid (n too small)",
        "refuted": "refuted",
        _UNAVAILABLE: "stats toolkit unavailable",
    }.get(verdict, verdict)
    return f"{pid}: {label} - {detail}"


def honest_scorecard(results: list) -> dict:
    """Assemble the power-aware scorecard for P2, P3 (+ a P5 descriptor).

    Each entry carries the underlying stats dict, CIs, and a one-line human
    summary that never overstates an underpowered or unavailable result.

    Returns:
        ``{"stats_available", "power_planning", "predictions": {P2,P3,P5},
        "summary": [str, ...]}``.
    """
    if not _STATS_OK:
        return {"stats_available": False,
                "power_planning": power_planning(),
                "predictions": {
                    "P2": _unavailable(),
                    "P3": _unavailable(),
                    "P5": _unavailable(),
                },
                "summary": [
                    _summary_line("P2", _UNAVAILABLE, "ztare.experiment_stats absent."),
                    _summary_line("P3", _UNAVAILABLE, "ztare.experiment_stats absent."),
                    _summary_line("P5", _UNAVAILABLE, "ztare.experiment_stats absent."),
                ]}

    p2 = p2_goodhart_verdict(results, arch="A")
    p3 = p3_gaming_verdict(results)

    # P5 - governance capture descriptor (not a stats verdict: capture_accepted is
    # a count that is structurally zero in most mock cells). Reported for honesty.
    arch_c = _by_arch(results).get("C", [])
    capture_total = sum(int(r.capture_accepted_total) for r in arch_c)
    p5 = {"arch": "C", "n_runs": len(arch_c),
          "capture_accepted_total": capture_total,
          "verdict": ("descriptive" if arch_c else "n/a"),
          "note": (f"{capture_total} capture proposal(s) accepted across "
                   f"{len(arch_c)} arch-C run(s); reported as a descriptor, not a "
                   f"powered test." if arch_c else "no arch-C runs present.")}

    summary = [
        _summary_line("P2", p2["verdict"], p2.get("note", "")),
        _summary_line("P3", p3["verdict"], p3.get("note", "")),
        _summary_line("P5", p5["verdict"], p5["note"]),
    ]

    return {"stats_available": True,
            "power_planning": power_planning(),
            "predictions": {"P2": p2, "P3": p3, "P5": p5},
            "summary": summary}
