"""Plotly figures for the session report.

Colours follow the eymeric.me brand: two spot inks (Prussian blue, vermilion) plus sumi grey
on washi paper.  Every trace carries a ``meta`` role; the page's JS re-inks traces from the
CSS tokens (``--c-<role>``) so light/dark mode stay on-brand.  Charts with several series use
at most the two inks plus grey as a context series, always with distinct marker shapes and
direct labels.  One measure per y-axis, never dual axes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal

from ..analysis.hr import ZONE_NAMES
from ..analyze import Result

# light-mode values; dark-mode equivalents live in the template's CSS tokens
C = {
    "hr": "#b83a20",  # vermilion: the heart
    "watts": "#1f4e79",  # prussian: the machine
    "spm": "#6c6254",  # sumi
    "breath": "#1f4e79",
    "alpha": "#1f4e79",
    "other": "#b83a20",
    "muted": "#8a8173",
}
# zones: cool prussian below the aerobic threshold -> neutral stone -> warm vermilion above
ZONE_COLORS = ["#a9c2da", "#5b87b5", "#b9ae9a", "#d4735a", "#b83a20"]
FONT = "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif"
MONO = "JetBrains Mono, ui-monospace, SF Mono, Menlo, monospace"


def _base(fig: go.Figure, height: int, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=48, r=14, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12),
        hovermode="x unified",
        hoverlabel=dict(font=dict(family=FONT, size=12)),
        showlegend=legend,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showline=False, zeroline=False, gridwidth=1, tickfont=dict(family=MONO, size=10))
    fig.update_yaxes(zeroline=False, gridwidth=1, tickfont=dict(family=MONO, size=10))
    return fig


def _titles_left(fig, n):
    for ann in fig.layout.annotations[:n]:
        ann.update(x=0, xanchor="left", font=dict(size=11, family=MONO))


def _rel(res: Result, t):
    return (np.asarray(t) - res.t_origin) / 60


def _shade_exercises(fig, res, rows="all"):
    for s in res.segments:
        if s.kind == "exercise":
            fig.add_vrect(exclude_empty_subplots=False, x0=_rel(res, s.t0), x1=_rel(res, s.t1), fillcolor=C["muted"], opacity=0.10,
                          line_width=0, layer="below", row=rows, col=1)


def timeline(res: Result) -> go.Figure:
    has_h10 = res.hr1s is not None
    rows = [("HEART RATE · bpm", "hr")] if has_h10 else []
    rows += [("POWER · W", "watts"), ("STROKE RATE · /min", "spm")]
    if res.breathing is not None:
        rows.append(("BREATHING · /min (rest only)", "breath"))
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=[r[0] for r in rows])
    idx = {k: i + 1 for i, (_, k) in enumerate(rows)}
    _shade_exercises(fig, res)
    # exercise labels on the first row
    for s in res.segments:
        if s.kind == "exercise" and s.duration_s > 150:
            fig.add_annotation(x=_rel(res, (s.t0 + s.t1) / 2), y=0.97, yref="y domain", text=s.label.split(",")[0][:22],
                               showarrow=False, yanchor="top", font=dict(size=10, color=C["muted"]), row=1, col=1)

    if has_h10:
        b = res.profile.bounds()
        for z in range(5):
            fig.add_hrect(exclude_empty_subplots=False, y0=b[z], y1=b[z + 1], fillcolor=ZONE_COLORS[z], opacity=0.13, line_width=0,
                          layer="below", row=idx["hr"], col=1)
        vt1 = res.summary.get("hr_at_alpha1_075")
        if vt1:
            fig.add_hline(exclude_empty_subplots=False, y=vt1, line=dict(color=C["alpha"], width=1, dash="dash"), row=idx["hr"], col=1,
                          annotation_text=f"aerobic threshold ≈ {vt1:.0f}", annotation_position="bottom right",
                          annotation_font=dict(size=10))
        h = res.hr1s
        z = res.profile.zone_of(h["hr"])
        fig.add_trace(go.Scatter(x=_rel(res, h["t"]), y=h["hr"], mode="lines", meta="hr", line=dict(color=C["hr"], width=2),
                                 customdata=np.array([ZONE_NAMES[max(k - 1, 0)] if k else "below Z1" for k in z]),
                                 hovertemplate="%{y:.0f} bpm · %{customdata}<extra></extra>"), row=idx["hr"], col=1)
        fig.update_yaxes(range=[max(40, h["hr"].min() - 10), b[5] + 2], row=idx["hr"], col=1)

    for p in res.session.pieces:
        st = p.strokes
        fig.add_trace(go.Scatter(x=_rel(res, st["t"]), y=st["watts"], mode="lines", meta="watts",
                                 line=dict(color=C["watts"], width=1.5),
                                 hovertemplate="%{y:.0f} W<extra></extra>"), row=idx["watts"], col=1)
        fig.add_trace(go.Scatter(x=_rel(res, st["t"]), y=st["spm"], mode="lines", meta="spm",
                                 line=dict(color=C["spm"], width=1.5),
                                 hovertemplate="%{y:.0f} spm (PM5)<extra></extra>"), row=idx["spm"], col=1)
    if res.activity is not None:
        a = res.activity[res.activity["rowing"] & (res.activity["spm"] > 14)]
        fig.add_trace(go.Scatter(x=_rel(res, a["t"]), y=a["spm"], mode="markers", meta="spm",
                                 marker=dict(color=C["spm"], size=4, opacity=0.35, symbol="circle-open"),
                                 hovertemplate="%{y:.1f} spm (chest ACC)<extra></extra>"), row=idx["spm"], col=1)
    fig.update_yaxes(range=[10, 32], row=idx["spm"], col=1)

    if res.breathing is not None:
        r = res.breathing.dropna(subset=["rate"])
        fig.add_trace(go.Scatter(x=_rel(res, r["t"]), y=r["rate"], mode="markers", meta="breath",
                                 marker=dict(color=C["breath"], size=7),
                                 hovertemplate="%{y:.0f} breaths/min<extra></extra>"), row=idx["breath"], col=1)
        fig.update_yaxes(range=[0, 40], row=idx["breath"], col=1)

    fig.update_xaxes(title_text="minutes" + (" since H10 start" if has_h10 else " (pieces laid end-to-end)"),
                     title_font=dict(size=11), row=len(rows), col=1)
    _base(fig, 150 + 165 * len(rows))
    fig.update_layout(hovermode="x")
    _titles_left(fig, len(rows))
    return fig


def zone_bar(zones_s: list[float], height: int = 90) -> go.Figure:
    tot = sum(zones_s) or 1
    fig = go.Figure()
    for z, v in enumerate(zones_s):
        fig.add_trace(go.Bar(y=[""], x=[v / 60], orientation="h", name=ZONE_NAMES[z],
                             marker=dict(color=ZONE_COLORS[z], line=dict(width=2, color="rgba(250,247,240,1)")),
                             text=[f"Z{z + 1} {100 * v / tot:.0f}%" if v / tot > 0.08 else ""],
                             textposition="inside", insidetextanchor="middle", textfont=dict(family=MONO, size=10),
                             hovertemplate=f"{ZONE_NAMES[z]}: %{{x:.1f}} min ({100 * v / tot:.0f}%)<extra></extra>"))
    _base(fig, height)
    fig.update_layout(barmode="stack", margin=dict(l=4, r=4, t=4, b=24), hovermode="closest", bargap=0.15)
    fig.update_xaxes(ticksuffix=" min")
    fig.update_yaxes(showticklabels=False, showgrid=False)
    return fig


def piece_detail(res: Result, seg) -> go.Figure | None:
    p = seg.piece
    has_hr = res.hr1s is not None
    rows = ([("POWER · W", "watts")] if p is not None else []) + ([("HEART RATE · bpm", "hr")] if has_hr else [])
    if not rows:
        return None
    fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=[r[0] for r in rows])
    x0 = seg.t0
    r = 1
    if p is not None:
        st = p.strokes
        fig.add_trace(go.Scatter(x=(st["t"] - x0) / 60, y=st["watts"], mode="lines+markers", meta="watts",
                                 line=dict(color=C["watts"], width=1.5), marker=dict(size=4),
                                 customdata=np.column_stack([st["spm"].fillna(0), st["distance"].diff().fillna(0)]),
                                 hovertemplate="%{y:.0f} W · %{customdata[0]:.0f} spm · %{customdata[1]:.1f} m/stroke<extra></extra>"),
                      row=1, col=1)
        tgt = seg.target.get("watts")
        if tgt:
            fig.add_hline(exclude_empty_subplots=False, y=tgt, line=dict(color=C["muted"], dash="dash", width=1), row=1, col=1,
                          annotation_text=f"target {tgt} W", annotation_position="top left", annotation_font=dict(size=10))
        r = 2
    if has_hr:
        h = res.hr1s[(res.hr1s["t"] >= seg.t0 - 30) & (res.hr1s["t"] <= seg.t1 + 90)]
        b = res.profile.bounds()
        for z in range(5):
            fig.add_hrect(exclude_empty_subplots=False, y0=b[z], y1=b[z + 1], fillcolor=ZONE_COLORS[z], opacity=0.13, line_width=0, layer="below", row=r, col=1)
        tz = seg.target.get("zone")
        if tz:
            fig.add_hrect(exclude_empty_subplots=False, y0=b[tz - 1], y1=b[tz], fillcolor=ZONE_COLORS[tz - 1], opacity=0.35, line_width=0, layer="below",
                          row=r, col=1, annotation_text=f"target Z{tz}", annotation_position="top left",
                          annotation_font=dict(size=10))
        fig.add_trace(go.Scatter(x=(h["t"] - x0) / 60, y=h["hr"], mode="lines", meta="hr", line=dict(color=C["hr"], width=2),
                                 hovertemplate="%{y:.0f} bpm<extra></extra>"), row=r, col=1)
        fig.add_vline(exclude_empty_subplots=False, x=seg.duration_s / 60, line=dict(color=C["muted"], width=1, dash="dot"), row=r, col=1)
        lo = h["hr"].min() - 8
        hi = max(h["hr"].max() + 8, b[tz] + 2 if tz else 0)
        fig.update_yaxes(range=[lo, hi], row=r, col=1)
    fig.update_xaxes(title_text="minutes into piece (dotted = end, then 90 s of recovery)", title_font=dict(size=11),
                     row=len(rows), col=1)
    _base(fig, 140 + 150 * len(rows))
    fig.update_layout(hovermode="x")
    _titles_left(fig, len(rows))
    return fig


def hr_power(res: Result) -> go.Figure | None:
    pts, fit = res.hr_power, res.hr_power_fit
    if pts is None or not len(pts):
        return None
    fig = go.Figure()
    z = res.recommendations.get("zones") if res.recommendations else None
    if z:
        fig.add_hrect(exclude_empty_subplots=False, y0=z["easy_hr"][0], y1=z["easy_hr"][1], fillcolor=ZONE_COLORS[1], opacity=0.15, line_width=0,
                      annotation_text="easy (below VT1)", annotation_position="top left", annotation_font=dict(size=10))
    symbols = ["circle", "diamond", "square", "triangle-up"]
    roles = ["watts", "other", "spm", "watts"]
    for i, (label, d) in enumerate(pts.groupby("label", sort=False)):
        fig.add_trace(go.Scatter(x=d["watts"], y=d["hr"], mode="markers", name=label, meta=roles[i % 4],
                                 marker=dict(size=9, symbol=symbols[i % 4], color=C[roles[i % 4]], line=dict(width=1, color="rgba(250,247,240,.9)")),
                                 customdata=d["minute_in_piece"],
                                 hovertemplate=f"<b>{label}</b><br>minute %{{customdata:.0f}}: %{{x:.0f}} W → %{{y:.0f}} bpm<extra></extra>"))
    if fit:
        xs = np.linspace(pts["watts"].min() - 10, pts["watts"].max() + 10, 20)
        fig.add_trace(go.Scatter(x=xs, y=fit["a"] + fit["b"] * xs, mode="lines", meta="muted", name="HR–power line",
                                 line=dict(color=C["muted"], width=1.5, dash="dash"),
                                 hovertemplate=f"fit: %{{x:.0f}} W → %{{y:.0f}} bpm (+{fit['b']:.2f} bpm per W)<extra></extra>"))
    vt1, wv = res.summary.get("hr_at_alpha1_075"), res.summary.get("watts_at_vt1")
    if vt1 and wv:
        fig.add_trace(go.Scatter(x=[wv], y=[vt1], mode="markers+text", meta="hr", name="aerobic threshold",
                                 marker=dict(size=14, symbol="x-thin", line=dict(width=3, color=C["hr"])),
                                 text=[f"  VT1 ≈ {wv:.0f} W @ {vt1:.0f} bpm"], textposition="middle right",
                                 textfont=dict(size=11), hovertemplate="aerobic threshold: %{x:.0f} W @ %{y:.0f} bpm<extra></extra>"))
    fig.update_xaxes(title_text="power (W, 1-min average)")
    fig.update_yaxes(title_text="heart rate (bpm)")
    _base(fig, 380, legend=True)
    fig.update_layout(hovermode="closest")
    return fig


def alpha1_timeline(res: Result) -> go.Figure | None:
    d = res.hrv
    if d is None or not len(d):
        return None
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.62, 0.38],
                        subplot_titles=["DFA α1 · 2-min windows", "HEART RATE · bpm"])
    _shade_exercises(fig, res)
    fig.add_hrect(exclude_empty_subplots=False, y0=0.75, y1=2.0, fillcolor=ZONE_COLORS[1], opacity=0.10, line_width=0, row=1, col=1,
                  annotation_text="below aerobic threshold", annotation_position="top left", annotation_font=dict(size=10))
    fig.add_hrect(exclude_empty_subplots=False, y0=0.5, y1=0.75, fillcolor=ZONE_COLORS[3], opacity=0.12, line_width=0, row=1, col=1,
                  annotation_text="between thresholds (grey zone)", annotation_position="bottom left", annotation_font=dict(size=10))
    fig.add_hrect(exclude_empty_subplots=False, y0=0, y1=0.5, fillcolor=ZONE_COLORS[4], opacity=0.12, line_width=0, row=1, col=1,
                  annotation_text="above anaerobic threshold", annotation_position="bottom left", annotation_font=dict(size=10))
    ok, bad = d[d["reliable"]], d[~d["reliable"]]
    fig.add_trace(go.Scatter(x=_rel(res, ok["t"]), y=ok["alpha1"], mode="markers", meta="alpha",
                             marker=dict(color=C["alpha"], size=6),
                             customdata=ok["hr"], hovertemplate="α1 %{y:.2f} · HR %{customdata:.0f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=_rel(res, bad["t"]), y=bad["alpha1"], mode="markers", meta="muted",
                             marker=dict(color=C["muted"], size=6, symbol="x-thin", line=dict(width=1.5, color=C["muted"])),
                             customdata=100 * bad["artefact"],
                             hovertemplate="rejected: %{customdata:.1f}% suspicious beats<extra></extra>"), row=1, col=1)
    h = res.hr1s
    fig.add_trace(go.Scatter(x=_rel(res, h["t"]), y=h["hr"], mode="lines", meta="hr", line=dict(color=C["hr"], width=1.5),
                             hovertemplate="%{y:.0f} bpm<extra></extra>"), row=2, col=1)
    vt1 = res.summary.get("hr_at_alpha1_075")
    if vt1:
        fig.add_hline(exclude_empty_subplots=False, y=vt1, line=dict(color=C["alpha"], width=1, dash="dash"), row=2, col=1)
    fig.update_yaxes(range=[0, 1.9], row=1, col=1)
    fig.update_xaxes(title_text="minutes since H10 start", title_font=dict(size=11), row=2, col=1)
    _base(fig, 440)
    fig.update_layout(hovermode="x")
    _titles_left(fig, 2)
    return fig


def alpha1_vs_hr(res: Result) -> go.Figure | None:
    d = res.hrv
    if d is None or not len(d):
        return None
    fig = go.Figure()
    fig.add_hrect(exclude_empty_subplots=False, y0=0.5, y1=0.75, fillcolor=ZONE_COLORS[3], opacity=0.12, line_width=0, layer="below")
    bad = d[~d["reliable"]]
    fig.add_trace(go.Scatter(x=bad["hr"], y=bad["alpha1"], mode="markers", name="rejected (noisy ECG)", meta="muted",
                             marker=dict(color=C["muted"], size=7, symbol="x-thin", line=dict(width=1.5, color=C["muted"])),
                             hovertemplate="HR %{x:.0f} · α1 %{y:.2f} (rejected)<extra></extra>"))
    ok = d[d["reliable"]]
    fig.add_trace(go.Scatter(x=ok["hr"], y=ok["alpha1"], mode="markers", name="2-min window", meta="alpha",
                             marker=dict(color=C["alpha"], size=8, opacity=0.75),
                             customdata=_rel(res, ok["t"]),
                             hovertemplate="HR %{x:.0f} · α1 %{y:.2f} · at %{customdata:.1f} min<extra></extra>"))
    fig.add_hline(exclude_empty_subplots=False, y=0.75, line=dict(color=C["muted"], dash="dash", width=1),
                  annotation_text="0.75 = aerobic threshold (VT1)", annotation_position="top left", annotation_font=dict(size=10))
    fig.add_hline(exclude_empty_subplots=False, y=0.5, line=dict(color=C["muted"], dash="dot", width=1),
                  annotation_text="0.5 = anaerobic threshold (VT2)", annotation_position="bottom left", annotation_font=dict(size=10))
    hr75 = res.summary.get("hr_at_alpha1_075")
    if hr75:
        fig.add_vline(exclude_empty_subplots=False, x=hr75, line=dict(color=C["hr"], width=1.5),
                      annotation_text=f"{hr75:.0f} bpm", annotation_position="top right")
    fig.update_xaxes(title_text="heart rate (bpm)")
    fig.update_yaxes(title_text="DFA α1", range=[0, 1.9])
    _base(fig, 360, legend=True)
    fig.update_layout(hovermode="closest")
    return fig


def breathing_trace(res: Result, t_center: float, span: float = 40.0) -> go.Figure:
    g, sig = res.resp_grid, res.resp_sig
    sel = (g >= t_center - span / 2) & (g <= t_center + span / 2)
    x = g[sel] - (t_center - span / 2)
    e = sig["edr"][sel]
    pk, _ = signal.find_peaks(e, distance=int(4 * 1.5), prominence=0.3 * np.std(e))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=[f"EDR · R-PEAK HEIGHT · {len(pk)} breaths in {span:.0f} s = {60 * len(pk) / span:.0f} /min",
                                        "RR INTERVAL (RSA) · ms"])
    fig.add_trace(go.Scatter(x=x, y=e, mode="lines", meta="breath", line=dict(color=C["breath"], width=2),
                             hovertemplate="%{y:.0f} µV<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x[pk], y=e[pk], mode="markers", meta="hr", marker=dict(color=C["hr"], size=9, symbol="triangle-down"),
                             hovertemplate="breath %{x:.1f} s<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=sig["rsa"][sel], mode="lines", meta="spm", line=dict(color=C["spm"], width=1.5),
                             hovertemplate="%{y:.1f} ms<extra></extra>"), row=2, col=1)
    fig.update_xaxes(title_text=f"seconds (window at {_rel(res, t_center):.1f} min)", title_font=dict(size=11), row=2, col=1)
    _base(fig, 340)
    fig.update_layout(hovermode="x")
    _titles_left(fig, 2)
    return fig


def breathing_recovery(res: Result) -> go.Figure | None:
    if not res.breath_recovery:
        return None
    fig = go.Figure()
    pre = next((s for s in res.segments if s.label == "Before start"), None)
    base = pre.metrics.get("breath_rate") if pre else None
    if base:
        fig.add_hline(exclude_empty_subplots=False, y=base, line=dict(color=C["muted"], dash="dot", width=1),
                      annotation_text=f"pre-session rest {base:.0f} /min", annotation_position="bottom right",
                      annotation_font=dict(size=10))
    symbols = ["circle", "diamond", "square", "triangle-up"]
    roles = ["watts", "other", "spm", "watts"]
    for i, (label, d) in enumerate(res.breath_recovery.items()):
        fig.add_trace(go.Scatter(x=d["since_s"] / 60, y=d["rate"], mode="lines+markers", name=label[:28], meta=roles[i % 4],
                                 line=dict(color=C[roles[i % 4]], width=1.5, dash="solid" if i < 2 else "dot"),
                                 marker=dict(size=7, symbol=symbols[i % 4], color=C[roles[i % 4]]),
                                 hovertemplate=f"<b>{label}</b><br>%{{x:.1f}} min after: %{{y:.0f}} /min<extra></extra>"))
    fig.update_xaxes(title_text="minutes after the effort ended")
    fig.update_yaxes(title_text="breaths / min", range=[0, 40])
    _base(fig, 340, legend=True)
    fig.update_layout(hovermode="closest")
    return fig


def coupling_bars(res: Result) -> go.Figure | None:
    ex = [s for s in res.segments if s.kind == "exercise" and s.metrics.get("breath_coupling_1") is not None]
    if not ex:
        return None
    labels = [s.label[:26] for s in ex]
    one = [100 * s.metrics["breath_coupling_1"] for s in ex]
    two = [100 * s.metrics.get("breath_coupling_2", 0) for s in ex]
    other = [max(0, 100 - a - b) for a, b in zip(one, two)]
    fig = go.Figure()
    for name, vals, role in [("1 breath / stroke", one, "watts"), ("2 breaths / stroke", two, "other"), ("not locked", other, "muted")]:
        fig.add_trace(go.Bar(y=labels, x=vals, name=name, orientation="h", meta=role,
                             marker=dict(color=C[role], line=dict(width=2, color="rgba(250,247,240,1)")),
                             text=[f"{v:.0f}%" if v >= 12 else "" for v in vals], textposition="inside",
                             textfont=dict(family=MONO, size=10, color="#faf7f0"),
                             hovertemplate="%{y}: %{x:.0f}% of windows " + name + "<extra></extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title_text="% of 32-s windows while rowing", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    _base(fig, 90 + 38 * len(ex), legend=True)
    fig.update_layout(hovermode="closest", margin=dict(l=170, r=14, t=64, b=40), legend=dict(y=1.02, yanchor="bottom"))
    return fig


def rowing_spectrum(res: Result, t_center: float, span: float = 32.0) -> tuple[go.Figure, float] | None:
    """EDR vs chest-motion spectra over one rowing window, and the EDR power share on stroke harmonics."""
    g, sig = res.resp_grid, res.resp_sig
    sel = (g >= t_center - span / 2) & (g < t_center + span / 2)
    if sel.sum() < 64:
        return None
    fs = 4.0
    f, pe = signal.welch(sig["edr"][sel], fs=fs, nperseg=sel.sum(), nfft=2048)
    _, pa = signal.welch(sig["acc"][sel], fs=fs, nperseg=sel.sum(), nfft=2048, axis=0)
    pa = pa.sum(axis=1)
    m = (f > 0.08) & (f < 1.1)
    act = res.activity.iloc[(res.activity["t"] - t_center).abs().argmin()]
    spm = float(act["spm"])
    on_stroke = np.zeros_like(f, dtype=bool)
    for k in (1, 2, 3):
        on_stroke |= np.abs(f * 60 - k * spm) < 2.0
    share = float(pe[m & on_stroke].sum() / pe[m].sum())
    fig = go.Figure()
    for k, nm in [(1, "stroke"), (2, "2× stroke")]:
        fig.add_vline(exclude_empty_subplots=False, x=k * spm, line=dict(color=C["muted"], width=1, dash="dot"),
                      annotation_text=f"{nm} {k * spm:.0f}", annotation_position="bottom right", annotation_font=dict(size=10))
    fig.add_trace(go.Scatter(x=f[m] * 60, y=pa[m] / pa[m].max(), mode="lines", name="chest motion (ACC)", meta="spm",
                             line=dict(color=C["spm"], width=1.5, dash="dash"),
                             hovertemplate="%{x:.1f} /min · %{y:.2f}<extra>ACC</extra>"))
    fig.add_trace(go.Scatter(x=f[m] * 60, y=pe[m] / pe[m].max(), mode="lines", name="ECG breathing signal (EDR)", meta="breath",
                             line=dict(color=C["breath"], width=2), fill="tozeroy",
                             hovertemplate="%{x:.1f} /min · %{y:.2f}<extra>EDR</extra>"))
    fig.update_xaxes(title_text=f"frequency (per minute) · 32 s of rowing at {_rel(res, t_center):.1f} min")
    fig.update_yaxes(title_text="relative power", showticklabels=False)
    _base(fig, 320, legend=True)
    fig.update_layout(hovermode="closest", legend=dict(y=1.02, yanchor="bottom"), margin=dict(t=48))
    return fig, share


def ecg_snippets(res: Result, t_rest: float, t_row: float, span: float = 6.0) -> go.Figure:
    b = res.beats
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=[f"AT REST · {_rel(res, t_rest):.1f} min", f"ROWING · {_rel(res, t_row):.1f} min"])
    for i, tc in enumerate((t_rest, t_row), start=1):
        sel = (b.ecg_t >= tc) & (b.ecg_t < tc + span)
        fig.add_trace(go.Scatter(x=b.ecg_t[sel] - tc, y=b.ecg[sel], mode="lines", meta="spm", line=dict(color=C["spm"], width=1),
                                 hoverinfo="skip"), row=i, col=1)
        pk = (b.t >= tc) & (b.t < tc + span)
        fig.add_trace(go.Scatter(x=b.t[pk] - tc, y=b.amp[pk], mode="markers", meta="hr",
                                 marker=dict(color=C["hr"], size=8, symbol="triangle-down"),
                                 hovertemplate="R-peak %{y:.0f} µV<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title_text="µV", row=i, col=1)
    fig.update_xaxes(title_text="seconds", row=2, col=1)
    _base(fig, 380)
    fig.update_layout(hovermode="closest")
    _titles_left(fig, 2)
    return fig


def progress(res: Result) -> go.Figure | None:
    """Cardio-progress markers across sessions (needs >= 2 sessions with H10 data)."""
    pts = []
    for m in res.history + [{"date": res.session.day.isoformat(), "summary": res.summary}]:
        s = m.get("summary", {})
        if s.get("watts_at_hr140") is not None:
            pts.append((m["date"], s.get("watts_at_hr140"), s.get("watts_at_vt1")))
    if len(pts) < 2:
        return None
    d, w140, wv = zip(*pts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d, y=w140, mode="lines+markers", name="watts at 140 bpm", meta="watts",
                             marker=dict(size=9), line=dict(color=C["watts"], width=2)))
    fig.add_trace(go.Scatter(x=d, y=wv, mode="lines+markers", name="watts at aerobic threshold", meta="other",
                             marker=dict(size=9, symbol="diamond"), line=dict(color=C["other"], width=2, dash="dot")))
    fig.update_yaxes(title_text="W")
    _base(fig, 300, legend=True)
    return fig
