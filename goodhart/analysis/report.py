"""Self-contained HTML dashboard for a set of Goodhart-on-the-Bridge runs.

Renders, with the Python standard library only (no matplotlib, no external CSS):

  * a header and a prominent caveat banner whenever any run is mock-backed
    (per the honesty note in `goodhart.runner.config`);
  * a boundary-integrity banner (green/red);
  * the seed-averaged Goodhart curve as an INLINE SVG (G_t vs lambda, one
    polyline per architecture);
  * a gaming-events bar chart as an INLINE SVG (by lambda);
  * the P1-P6 scorecard table with colour-coded verdicts;
  * a per-run summary table; and
  * an expandable per-epoch table for every run.

The HTML never depends on a successful plot library import, so the report always
generates. A matplotlib PNG may *additionally* be written by `write_report`, but
that path is fully guarded by try/except and is optional.

Determinism: no clock, uuid, or unseeded randomness. The output path and title
are supplied by the caller.
"""

from __future__ import annotations

import json
from html import escape

from .curve import (_by_lam, _post_feedback_gaps, evaluate_predictions,
                    goodhart_curve)

# Stable colour per architecture for both SVG charts and legends.
_ARCH_COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
_ARCH_NAMES = {"A": "A typed-static", "B": "B free-form", "C": "C typed-governed"}
_VERDICT_COLORS = {
    "supported": "#1a7f37", "refuted": "#cf222e",
    "inconclusive": "#9a6700", "n/a": "#57606a",
}


# -- small helpers ---------------------------------------------------------
def _esc(x) -> str:
    """HTML-escape any value as text."""
    return escape(str(x))


def _fmt(x, nd: int = 3) -> str:
    """Format a number to `nd` decimals, passing non-numbers through escaped."""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(x)


def _all_lams(results: list) -> list[float]:
    """Sorted unique lambda values across all runs."""
    return sorted({float(r.config["lam"]) for r in results})


# -- inline SVG: the Goodhart curve ----------------------------------------
def _svg_curve(curve: dict, results: list) -> str:
    """Render the seed-averaged Goodhart curve (G_t vs lambda) as inline SVG.

    One polyline per architecture using its terminal-mean gap; axis labels, a
    legend, and a marker at every (lambda, G) point. Degrades to a friendly
    message if there is nothing to plot.
    """
    W, H = 640, 360
    ml, mr, mt, mb = 64, 150, 28, 52
    pw, ph = W - ml - mr, H - mt - mb

    lams = _all_lams(results)
    all_g = [g for rec in curve.values() for g in rec["terminal_mean"]]
    if not lams or not all_g:
        return ('<svg width="640" height="80" role="img">'
                '<text x="12" y="44">No curve data to plot.</text></svg>')

    lam_lo, lam_hi = min(lams), max(lams)
    g_lo, g_hi = 0.0, max(all_g)
    g_hi = g_hi if g_hi > 0 else 1.0
    lam_span = (lam_hi - lam_lo) or 1.0

    def x_of(lam: float) -> float:
        return ml + (lam - lam_lo) / lam_span * pw

    def y_of(g: float) -> float:
        return mt + (1.0 - (g - g_lo) / (g_hi - g_lo)) * ph

    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
             'role="img" aria-label="Goodhart curve: oversight gap vs lambda" '
             'style="font-family:system-ui,sans-serif;font-size:12px">']
    # plot frame
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                 'fill="#fbfbfd" stroke="#d0d7de"/>')
    # y gridlines + labels
    for i in range(5):
        gv = g_lo + (g_hi - g_lo) * i / 4
        yy = y_of(gv)
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
                     'stroke="#eaeef2"/>')
        parts.append(f'<text x="{ml - 8:.0f}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'fill="#57606a">{gv:.2f}</text>')
    # x ticks + labels
    for lam in lams:
        xx = x_of(lam)
        parts.append(f'<line x1="{xx:.1f}" y1="{mt + ph}" x2="{xx:.1f}" '
                     f'y2="{mt + ph + 5}" stroke="#57606a"/>')
        parts.append(f'<text x="{xx:.1f}" y="{mt + ph + 20}" text-anchor="middle" '
                     f'fill="#57606a">{lam:g}</text>')
    # axis titles
    parts.append(f'<text x="{ml + pw / 2:.0f}" y="{H - 8}" text-anchor="middle" '
                 'fill="#24292f">pressure &#955; (lambda)</text>')
    parts.append(f'<text transform="translate(16,{mt + ph / 2:.0f}) rotate(-90)" '
                 'text-anchor="middle" fill="#24292f">terminal oversight gap '
                 'G_t</text>')
    # one polyline + markers per architecture
    for j, (arch, rec) in enumerate(sorted(curve.items())):
        color = _ARCH_COLORS.get(arch, "#8250df")
        pts = [(x_of(l), y_of(g)) for l, g in zip(rec["lams"], rec["terminal_mean"])]
        if len(pts) >= 2:
            poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
            parts.append(f'<polyline points="{poly}" fill="none" '
                         f'stroke="{color}" stroke-width="2.5"/>')
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        ly = mt + 6 + j * 20
        parts.append(f'<rect x="{ml + pw + 16}" y="{ly}" width="14" height="4" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{ml + pw + 34}" y="{ly + 6}" fill="#24292f">'
                     f'{_esc(_ARCH_NAMES.get(arch, arch))}</text>')
    parts.append('</svg>')
    return "".join(parts)


# -- inline SVG: gaming-events bar chart ------------------------------------
def _svg_gaming_bars(results: list) -> str:
    """Render mean gaming-event count by lambda (averaged across all runs at each
    lambda) as an inline-SVG bar chart."""
    W, H = 640, 300
    ml, mr, mt, mb = 64, 24, 28, 52
    pw, ph = W - ml - mr, H - mt - mb

    lam_groups = _by_lam(results)
    lams = sorted(lam_groups.keys())
    if not lams:
        return ('<svg width="640" height="80" role="img">'
                '<text x="12" y="44">No gaming data to plot.</text></svg>')
    means = []
    for lam in lams:
        cell = lam_groups[lam]
        means.append(sum(r.gaming_event_count for r in cell) / len(cell))
    top = max(means) or 1.0

    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
             'aria-label="Gaming events by lambda" '
             'style="font-family:system-ui,sans-serif;font-size:12px">']
    parts.append(f'<rect x="{ml}" y="{mt}" width="{pw}" height="{ph}" '
                 'fill="#fbfbfd" stroke="#d0d7de"/>')
    for i in range(5):
        gv = top * i / 4
        yy = mt + (1 - i / 4) * ph
        parts.append(f'<line x1="{ml}" y1="{yy:.1f}" x2="{ml + pw}" y2="{yy:.1f}" '
                     'stroke="#eaeef2"/>')
        parts.append(f'<text x="{ml - 8}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'fill="#57606a">{gv:.1f}</text>')
    n = len(lams)
    slot = pw / n
    bw = slot * 0.6
    for i, (lam, mval) in enumerate(zip(lams, means)):
        bh = (mval / top) * ph if top else 0.0
        bx = ml + i * slot + (slot - bw) / 2
        by = mt + ph - bh
        parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" '
                     f'height="{bh:.1f}" fill="#8250df"/>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{by - 4:.1f}" '
                     f'text-anchor="middle" fill="#24292f">{mval:.1f}</text>')
        parts.append(f'<text x="{bx + bw / 2:.1f}" y="{mt + ph + 20}" '
                     f'text-anchor="middle" fill="#57606a">{lam:g}</text>')
    parts.append(f'<text x="{ml + pw / 2:.0f}" y="{H - 8}" text-anchor="middle" '
                 'fill="#24292f">pressure &#955; (lambda)</text>')
    parts.append(f'<text transform="translate(16,{mt + ph / 2:.0f}) rotate(-90)" '
                 'text-anchor="middle" fill="#24292f">mean gaming events</text>')
    parts.append('</svg>')
    return "".join(parts)


# -- HTML section builders -------------------------------------------------
def _scorecard_rows(predictions: list) -> str:
    """Table rows for the P1-P6 scorecard with colour-coded verdicts."""
    rows = []
    for p in predictions:
        color = _VERDICT_COLORS.get(p["verdict"], "#57606a")
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(p["id"])}</td>'
            f'<td>{_esc(p["claim"])}</td>'
            f'<td><span class="badge" style="background:{color}">'
            f'{_esc(p["verdict"])}</span></td>'
            f'<td>{_esc(p["evidence"])}</td>'
            "</tr>")
    return "".join(rows)


def _summary_rows(results: list) -> str:
    """Table rows for the per-run summary (label, terminal G, gaming, capture,
    boundary)."""
    rows = []
    for r in results:
        b_ok = r.boundary_all_clear
        b_html = (f'<span style="color:#1a7f37">clear</span>' if b_ok
                  else '<span style="color:#cf222e">LEAK</span>')
        rows.append(
            "<tr>"
            f'<td class="mono">{_esc(r.config["label"])}</td>'
            f'<td>{_fmt(r.terminal_gap)}</td>'
            f'<td>{_esc(r.gaming_event_count)}</td>'
            f'<td>{_esc(r.capture_accepted_total)}</td>'
            f'<td>{b_html}</td>'
            "</tr>")
    return "".join(rows)


def _epoch_blocks(results: list) -> str:
    """One <details> block per run with its per-epoch table (rating, gap,
    attribution, labels)."""
    blocks = []
    for r in results:
        head = (
            "<tr><th>epoch</th><th>eff &#955;</th><th>rating</th><th>gap</th>"
            "<th>attributed</th><th>schema</th><th>boundary</th><th>labels</th></tr>")
        body = []
        for e in r.epochs:
            attr = "" if e.gaming_attributed is None else _fmt(e.gaming_attributed, 4)
            bclear = "clear" if e.boundary_clear else "LEAK"
            labels = ", ".join(str(x) for x in e.labels)
            body.append(
                "<tr>"
                f"<td>{_esc(e.epoch)}</td><td>{_fmt(e.eff_lam, 2)}</td>"
                f"<td>{_fmt(e.rating)}</td><td>{_fmt(e.gap)}</td>"
                f"<td>{_esc(attr)}</td><td class='mono'>{_esc(e.schema_version)}</td>"
                f"<td>{_esc(bclear)}</td><td>{_esc(labels)}</td>"
                "</tr>")
        blocks.append(
            "<details><summary>" + _esc(r.config["label"]) + "</summary>"
            "<table class='grid'>" + head + "".join(body) + "</table></details>")
    return "".join(blocks)


def _per_run_summaries(results: list) -> list:
    """Compact JSON-friendly per-run records for the sidecar .json file."""
    out = []
    for r in results:
        out.append({
            "label": r.config["label"],
            "architecture": r.config["architecture"],
            "lam": r.config["lam"],
            "seed": r.config["seed"],
            "backend": r.config["backend"],
            "terminal_gap": r.terminal_gap,
            "post_feedback_mean_gap": (
                sum(_post_feedback_gaps(r)) / len(_post_feedback_gaps(r))
                if _post_feedback_gaps(r) else 0.0),
            "gaming_event_count": r.gaming_event_count,
            "capture_accepted_total": r.capture_accepted_total,
            "boundary_all_clear": r.boundary_all_clear,
            "gt_commit": r.gt_commit,
            "ratings": r.ratings,
            "gaps": r.gaps,
        })
    return out


# -- top-level rendering ---------------------------------------------------
def render_html(results: list, *, title: str = "Goodhart on the Bridge — Results") -> str:
    """Render a complete, self-contained HTML dashboard string for `results`.

    The returned string embeds its own CSS and inline-SVG charts; it has no
    external dependencies and always renders (even on empty input). Surfaces the
    mock-stipulation caveat prominently whenever any run is mock-backed.

    Args:
        results: a list of `RunResult` objects to summarise.
        title: the page/header title (caller-supplied; not derived from a clock).

    Returns:
        A full HTML document as a `str`.
    """
    curve = goodhart_curve(results)
    predictions = evaluate_predictions(results)["predictions"]

    any_mock = any(r.config.get("backend") == "mock" for r in results)
    all_clear = bool(results) and all(r.boundary_all_clear for r in results)
    lams = _all_lams(results)
    lam_str = ", ".join(f"{l:g}" for l in lams) if lams else "(none)"

    mock_banner = ""
    if any_mock:
        mock_banner = (
            '<div class="banner warn"><strong>MOCK BACKEND &mdash; stipulated, not '
            'measured.</strong> One or more runs use <code>backend="mock"</code>. '
            'The mock makes export coverage and gaming propensity explicit functions '
            'of &#955; (a <em>stipulated behavioral model</em>, per the honesty note '
            'in <code>config.py</code>). Any Goodhart curve shown here reflects that '
            'stipulation, <strong>not a measurement</strong> of an inner world&#39;s '
            'actual response. Only a real-backend run measures endogenous behavior.'
            '</div>')

    if not results:
        boundary_banner = ('<div class="banner neutral">No runs provided &mdash; '
                           'nothing to verify.</div>')
    elif all_clear:
        boundary_banner = ('<div class="banner ok"><strong>Boundary integrity: '
                           'CLEAR.</strong> All runs satisfy the metadata-blind '
                           'boundary (D7).</div>')
    else:
        leaks = [r.config["label"] for r in results if not r.boundary_all_clear]
        boundary_banner = ('<div class="banner err"><strong>Boundary integrity: '
                           'LEAK.</strong> Boundary not clear for: '
                           + _esc(", ".join(leaks)) + ".</div>")

    css = """
      :root{color-scheme:light}
      body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;
           background:#f6f8fa;color:#24292f;line-height:1.45}
      .wrap{max-width:960px;margin:0 auto;padding:24px}
      h1{font-size:24px;margin:0 0 4px} h2{font-size:18px;margin:28px 0 10px;
         border-bottom:1px solid #d0d7de;padding-bottom:6px}
      .sub{color:#57606a;margin:0 0 16px}
      .banner{padding:12px 16px;border-radius:8px;margin:12px 0;border:1px solid}
      .banner.ok{background:#dafbe1;border-color:#2da44e;color:#0a3d1a}
      .banner.err{background:#ffebe9;border-color:#cf222e;color:#82071e}
      .banner.warn{background:#fff8c5;border-color:#d4a72c;color:#5c4813}
      .banner.neutral{background:#eef1f4;border-color:#d0d7de;color:#57606a}
      .card{background:#fff;border:1px solid #d0d7de;border-radius:8px;padding:16px;
            margin:12px 0;overflow-x:auto}
      table{border-collapse:collapse;width:100%;font-size:13px}
      th,td{text-align:left;padding:6px 10px;border-bottom:1px solid #eaeef2}
      th{background:#f6f8fa}
      table.grid th,table.grid td{border:1px solid #eaeef2}
      .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
      .badge{color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;
             text-transform:uppercase;letter-spacing:.02em}
      details{margin:6px 0} summary{cursor:pointer;font-weight:600;padding:4px 0}
      footer{color:#57606a;font-size:12px;margin-top:28px}
    """

    html = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{_esc(title)}</title><style>{css}</style></head><body><div class='wrap'>",
        f"<h1>{_esc(title)}</h1>",
        f"<p class='sub'>{len(results)} run(s) &middot; &#955; grid: "
        f"{_esc(lam_str)} &middot; architectures: "
        f"{_esc(', '.join(sorted(curve.keys())) or '(none)')}</p>",
        mock_banner,
        boundary_banner,
        "<h2>Goodhart curve &mdash; terminal G_t vs &#955;</h2>",
        "<div class='card'>" + _svg_curve(curve, results) + "</div>",
        "<h2>Gaming events by &#955;</h2>",
        "<div class='card'>" + _svg_gaming_bars(results) + "</div>",
        "<h2>Prediction scorecard (P1&ndash;P6)</h2>",
        "<div class='card'><table><tr><th>ID</th><th>Claim</th><th>Verdict</th>"
        "<th>Evidence</th></tr>" + _scorecard_rows(predictions) + "</table></div>",
        "<h2>Per-run summary</h2>",
        "<div class='card'><table><tr><th>Label</th><th>Terminal G</th>"
        "<th>Gaming events</th><th>Capture total</th><th>Boundary</th></tr>"
        + _summary_rows(results) + "</table></div>",
        "<h2>Per-epoch detail</h2>",
        "<div class='card'>" + _epoch_blocks(results) + "</div>",
        "<footer>Self-contained report &middot; inline SVG, stdlib only. "
        "G_t is the exactly-computed oversight gap; lower is better.</footer>",
        "</div></body></html>",
    ]
    return "".join(html)


def write_report(results: list, path: str,
                 *, title: str = "Goodhart on the Bridge — Results") -> str:
    """Write the HTML dashboard to `path` and a sidecar `<path>.json`.

    The sidecar JSON contains the raw seed-averaged curve, the P1-P6 prediction
    verdicts, and per-run summaries (`json.dumps(..., default=str)`). Optionally
    also writes a matplotlib PNG next to the report, but only if matplotlib
    imports successfully; the HTML never depends on it.

    Args:
        results: a list of `RunResult` objects.
        path: the output HTML file path (supplied by the caller, not the clock).
        title: the page/header title.

    Returns:
        The `path` that was written.
    """
    html = render_html(results, title=title)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)

    sidecar = {
        "title": title,
        "curve": goodhart_curve(results),
        "predictions": evaluate_predictions(results)["predictions"],
        "runs": _per_run_summaries(results),
    }
    with open(path + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    # Optional, fully-guarded PNG. Never affects the HTML/JSON deliverables.
    try:  # pragma: no cover - depends on an optional dependency
        _write_png(results, path)
    except Exception:
        pass

    return path


def _write_png(results: list, path: str) -> None:  # pragma: no cover - optional
    """Best-effort matplotlib PNG of the Goodhart curve, guarded by the caller."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = goodhart_curve(results)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for arch, rec in sorted(curve.items()):
        ax.plot(rec["lams"], rec["terminal_mean"], marker="o",
                color=_ARCH_COLORS.get(arch, None),
                label=_ARCH_NAMES.get(arch, arch))
    ax.set_xlabel("pressure lambda")
    ax.set_ylabel("terminal oversight gap G_t")
    ax.set_title("Goodhart curve")
    ax.legend()
    fig.tight_layout()
    png_path = path.rsplit(".", 1)[0] + ".png" if "." in path else path + ".png"
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
