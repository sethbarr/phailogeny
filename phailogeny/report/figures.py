"""Standalone SVG tree figures for the README (no plotting library; attributes only, no CSS,
so GitHub renders them as images)."""

from __future__ import annotations

import math
from html import escape

COLOURS = ["#2a78d6", "#d9542b", "#1f9e6e", "#a05bd0", "#c9a227", "#d6457f", "#3aa6b8", "#7a5c3e", "#5b8c2a", "#8a8a8a"]


def colour_map(sources: list[str]) -> dict[str, str]:
    return {src: COLOURS[i % len(COLOURS)] for i, src in enumerate(sorted(set(sources)))}


def _depths(tree: object) -> dict[int, float]:
    depth: dict[int, float] = {}
    for node in tree.preorder():
        if node.parent is None:
            depth[id(node)] = 0.0
        else:
            depth[id(node)] = depth[id(node.parent)] + max(float(node.length or 0.0), 0.0)
    return depth


def _legend(colours: dict[str, str], x: float, y: float, counts: dict[str, int]) -> str:
    items = []
    for k, (src, col) in enumerate(colours.items()):
        yy = y + k * 18
        items.append(
            f'<rect x="{x}" y="{yy - 9}" width="11" height="11" rx="2" fill="{col}"/>'
            f'<text x="{x + 17}" y="{yy}" font-family="Helvetica,Arial,sans-serif" font-size="12" fill="#333">'
            f"{escape(src)} ({counts.get(src, 0)})</text>"
        )
    return "".join(items)


def radial_svg(tree: object, source_of: dict[str, str], size: int = 900, title: str = "") -> str:
    """Circular phylogram: tips evenly spread by angle, radius = path length from the root."""
    depth = _depths(tree)
    tips = list(tree.tips())
    max_depth = max(depth[id(t)] for t in tips) or 1.0
    radius = size * 0.42
    cx = cy = size / 2
    angle: dict[int, float] = {}
    for k, tip in enumerate(tips):
        angle[id(tip)] = 2 * math.pi * k / len(tips)
    for node in tree.postorder():
        if node.children:
            angles = [angle[id(c)] for c in node.children]
            angle[id(node)] = (min(angles) + max(angles)) / 2

    def point(a: float, r: float) -> tuple[float, float]:
        return cx + r * math.cos(a), cy + r * math.sin(a)

    parts = []
    for node in tree.preorder():
        if not node.children:
            continue
        r = depth[id(node)] / max_depth * radius
        angles = [angle[id(c)] for c in node.children]
        a0, a1 = min(angles), max(angles)
        x0, y0 = point(a0, r)
        x1, y1 = point(a1, r)
        large = 1 if a1 - a0 > math.pi else 0
        parts.append(f'<path d="M{x0:.1f},{y0:.1f} A{r:.1f},{r:.1f} 0 {large} 1 {x1:.1f},{y1:.1f}" fill="none" stroke="#999" stroke-width="0.5"/>')
        for child in node.children:
            xa, ya = point(angle[id(child)], r)
            xb, yb = point(angle[id(child)], depth[id(child)] / max_depth * radius)
            parts.append(f'<line x1="{xa:.1f}" y1="{ya:.1f}" x2="{xb:.1f}" y2="{yb:.1f}" stroke="#999" stroke-width="0.5"/>')
    colours = colour_map(list(source_of.values()))
    for tip in tips:
        x, y = point(angle[id(tip)], depth[id(tip)] / max_depth * radius + 6)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="{colours.get(source_of.get(str(tip.name), ""), "#555")}"/>')
    counts: dict[str, int] = {}
    for src in source_of.values():
        counts[src] = counts.get(src, 0) + 1
    width = size + 230
    head = (
        f'<text x="16" y="26" font-family="Helvetica,Arial,sans-serif" font-size="16" font-weight="bold" fill="#222">{escape(title)}</text>'
        if title else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{size}" viewBox="0 0 {width} {size}">'
        f'<rect width="{width}" height="{size}" fill="#ffffff"/>{head}{"".join(parts)}'
        f"{_legend(colours, size + 10, 60, counts)}</svg>"
    )


def rect_svg(tree: object, source_of: dict[str, str], title: str = "", row: int = 16) -> str:
    """Rectangular phylogram with labelled tips, for a readable clade."""
    depth = _depths(tree)
    tips = list(tree.tips())
    max_depth = max(depth[id(t)] for t in tips) or 1.0
    tree_width, label_width, top = 420, 360, 40
    y: dict[int, float] = {id(t): top + k * row for k, t in enumerate(tips)}
    for node in tree.postorder():
        if node.children:
            ys = [y[id(c)] for c in node.children]
            y[id(node)] = (min(ys) + max(ys)) / 2
    colours = colour_map(list(source_of.values()))

    def x_of(node: object) -> float:
        return 12 + depth[id(node)] / max_depth * tree_width

    parts = []
    for node in tree.preorder():
        if node.children:
            ys = [y[id(c)] for c in node.children]
            parts.append(f'<line x1="{x_of(node):.1f}" y1="{min(ys):.1f}" x2="{x_of(node):.1f}" y2="{max(ys):.1f}" stroke="#555" stroke-width="1"/>')
            for child in node.children:
                parts.append(f'<line x1="{x_of(node):.1f}" y1="{y[id(child)]:.1f}" x2="{x_of(child):.1f}" y2="{y[id(child)]:.1f}" stroke="#555" stroke-width="1"/>')
        else:
            src = source_of.get(str(node.name), "")
            parts.append(
                f'<text x="{x_of(node) + 5:.1f}" y="{y[id(node)] + 4:.1f}" font-family="Menlo,Consolas,monospace" font-size="12" '
                f'fill="{colours.get(src, "#555")}">{escape(str(node.name).replace("_", " "))}</text>'
            )
    height = top + len(tips) * row + 10
    width = tree_width + label_width
    head = (
        f'<text x="12" y="22" font-family="Helvetica,Arial,sans-serif" font-size="15" font-weight="bold" fill="#222">{escape(title)}</text>'
        if title else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>{head}{"".join(parts)}</svg>'
    )


def ranking_svg(
    ranking: list[dict],
    title: str,
    subtitle: str,
    reference: str | None = "control/no-prompt",
    reference_value: float | None = None,
    reference_label: str = "no-prompt control",
    value_key: str = "pass_rate",
    ci_key: str = "ci95",
    axis_label: str = "Share of bug-fix tasks whose tests pass (95% bootstrap interval over tasks)",
    fmt=lambda v: f"{v * 100:.0f}%",
    step: float = 0.1,
    note_key: str | None = None,
    palette: dict | None = None,
) -> str:
    """Dot-and-interval chart: pass rate per agent with its 95% CI, best at the top.

    Agents are blue dots, controls orange diamonds; identity is also in the text label. A dashed
    line marks the reference control's score. The x-axis starts where the data does (a dot plot,
    not a bar, so it need not start at zero) and the range is printed on the axis.
    """
    theme = {"agent": "#2160A8", "control": "#B04E17", "ink": "#17212B", "muted": "#5C6570",
             "grid": "#E3DFD6", "background": "#FFFFFF", "font": "Helvetica,Arial,sans-serif"} | (palette or {})
    agent_colour, control_colour = theme["agent"], theme["control"]
    ink, muted, grid = theme["ink"], theme["muted"], theme["grid"]
    font = theme["font"]
    label_w, plot_w, right_w, row, top = 470, 760, 90, 34, 104
    width = label_w + plot_w + right_w
    height = top + len(ranking) * row + 64
    lo = math.floor(min(r[ci_key][0] for r in ranking) / step) * step
    if value_key == "pass_rate":
        lo = max(0.0, lo)
    hi = math.ceil(max(r[ci_key][1] for r in ranking) / step) * step
    if value_key == "pass_rate":
        hi = 1.0

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot_w

    parts = [
        f'<rect width="{width}" height="{height}" fill="{theme["background"]}"/>',
        f'<text x="16" y="34" font-family="{font}" font-size="20" font-weight="bold" fill="{ink}">{escape(title)}</text>',
        f'<text x="16" y="60" font-family="{font}" font-size="14" fill="{muted}">{escape(subtitle)}</text>',
    ]
    ticks = [round(lo + k * step, 6) for k in range(int(round((hi - lo) / step)) + 1)]
    axis_y = top + len(ranking) * row + 8
    for t in ticks:
        parts.append(f'<line x1="{x(t):.1f}" y1="{top - 14}" x2="{x(t):.1f}" y2="{axis_y}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{axis_y + 20}" font-family="{font}" font-size="13" fill="{muted}" text-anchor="middle">{fmt(t)}</text>')
    parts.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{axis_y + 44}" font-family="{font}" font-size="13" fill="{muted}" text-anchor="middle">{escape(axis_label)}</text>')
    ref = next((r for r in ranking if r["agent_id"] == reference), None) if reference else None
    ref_value = ref[value_key] if ref else reference_value
    if ref_value is not None:
        rx = x(ref_value)
        parts.append(f'<line x1="{rx:.1f}" y1="{top - 14}" x2="{rx:.1f}" y2="{axis_y}" stroke="{control_colour}" stroke-width="1.5" stroke-dasharray="5 4"/>')
        near_right = rx > label_w + plot_w - 160
        anchor = ' text-anchor="end"' if near_right else ""
        tx = rx - 6 if near_right else rx + 6
        parts.append(f'<text x="{tx:.1f}" y="{top - 20}" font-family="{font}" font-size="13" fill="{muted}"{anchor}>{escape(reference_label)}</text>')
    for k, r in enumerate(ranking):
        cy = top + k * row + row / 2
        control = r["agent_id"].startswith("control/")
        colour = control_colour if control else agent_colour
        parts.append(f'<text x="{label_w - 14}" y="{cy + 5:.1f}" font-family="{font}" font-size="15" fill="{ink}" text-anchor="end"{" font-style=\"italic\"" if control else ""}>{escape(r["agent_id"])}</text>')
        parts.append(f'<line x1="{x(r[ci_key][0]):.1f}" y1="{cy:.1f}" x2="{x(r[ci_key][1]):.1f}" y2="{cy:.1f}" stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        px = x(r[value_key])
        if control:
            parts.append(f'<path d="M{px:.1f} {cy - 8:.1f} L{px + 8:.1f} {cy:.1f} L{px:.1f} {cy + 8:.1f} L{px - 8:.1f} {cy:.1f} Z" fill="{colour}" stroke="{theme["background"]}" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{px:.1f}" cy="{cy:.1f}" r="6.5" fill="{colour}" stroke="{theme["background"]}" stroke-width="2"/>')
        parts.append(f'<text x="{label_w + plot_w + 14}" y="{cy + 5:.1f}" font-family="{font}" font-size="15" fill="{muted}">{escape(r[note_key]) if note_key else fmt(r[value_key])}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(parts)}</svg>'


def slide_ranking_svg(
    ranking: list[dict],
    value_key: str,
    ci_key: str,
    axis_label: str,
    fmt,
    step: float,
    width: int = 1664,
    height: int = 700,
    font_px: int = 24,
    reference_value: float | None = None,
    reference_label: str = "",
    palette: dict | None = None,
    lo: float | None = None,
    hi: float | None = None,
) -> str:
    """A dot-and-interval chart sized to a slide's content box, so text renders at `font_px`
    on the slide itself (no title: the slide carries it). Agents are dots, controls diamonds."""
    theme = SPECIMEN | (palette or {})
    font, ink, muted, grid = theme["font"], theme["ink"], theme["muted"], theme["grid"]
    label_w = int(max(len(r["agent_id"]) for r in ranking) * font_px * 0.55) + 24
    value_w = int(font_px * 4.2)
    top, bottom = 34, int(font_px * 3.2)
    row = (height - top - bottom) / len(ranking)
    plot_w = width - label_w - value_w - 16
    lo = lo if lo is not None else math.floor(min(r[ci_key][0] for r in ranking) / step) * step
    hi = hi if hi is not None else math.ceil(max(r[ci_key][1] for r in ranking) / step) * step

    def x(v: float) -> float:
        return label_w + (v - lo) / (hi - lo) * plot_w

    axis_y = top + len(ranking) * row + 6
    parts = [f'<rect width="{width}" height="{height}" fill="{theme["background"]}"/>']
    for k in range(int(round((hi - lo) / step)) + 1):
        t = lo + k * step
        parts.append(f'<line x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{axis_y:.1f}" stroke="{grid}" stroke-width="1"/>')
        parts.append(f'<text x="{x(t):.1f}" y="{axis_y + font_px + 4:.1f}" font-family="{font}" font-size="{font_px - 2}" fill="{muted}" text-anchor="middle">{fmt(t)}</text>')
    parts.append(f'<text x="{label_w + plot_w / 2:.1f}" y="{axis_y + 2 * font_px + 14:.1f}" font-family="{font}" font-size="{font_px - 2}" fill="{muted}" text-anchor="middle">{escape(axis_label)}</text>')
    if reference_value is not None:
        rx = x(reference_value)
        near_right = rx > label_w + plot_w - 260
        anchor = ' text-anchor="end"' if near_right else ""
        parts.append(f'<line x1="{rx:.1f}" y1="{top - 8}" x2="{rx:.1f}" y2="{axis_y:.1f}" stroke="{theme["control"]}" stroke-width="2" stroke-dasharray="6 5"/>')
        parts.append(f'<text x="{rx - 8 if near_right else rx + 8:.1f}" y="{top - 12}" font-family="{font}" font-size="{font_px - 4}" fill="{muted}"{anchor}>{escape(reference_label)}</text>')
    for k, r in enumerate(ranking):
        cy = top + k * row + row / 2
        control = r["agent_id"].startswith("control/")
        colour = theme["control"] if control else theme["agent"]
        style = ' font-style="italic"' if control else ""
        parts.append(f'<text x="{label_w - 16}" y="{cy + font_px * 0.35:.1f}" font-family="{font}" font-size="{font_px}" fill="{ink}" text-anchor="end"{style}>{escape(r["agent_id"])}</text>')
        parts.append(f'<line x1="{x(r[ci_key][0]):.1f}" y1="{cy:.1f}" x2="{x(r[ci_key][1]):.1f}" y2="{cy:.1f}" stroke="{colour}" stroke-width="3" stroke-linecap="round"/>')
        px, m = x(r[value_key]), font_px * 0.42
        if control:
            parts.append(f'<path d="M{px:.1f} {cy - m:.1f} L{px + m:.1f} {cy:.1f} L{px:.1f} {cy + m:.1f} L{px - m:.1f} {cy:.1f} Z" fill="{colour}" stroke="{theme["background"]}" stroke-width="2"/>')
        else:
            parts.append(f'<circle cx="{px:.1f}" cy="{cy:.1f}" r="{m * 0.85:.1f}" fill="{colour}" stroke="{theme["background"]}" stroke-width="2"/>')
        parts.append(f'<text x="{label_w + plot_w + 16}" y="{cy + font_px * 0.35:.1f}" font-family="{font}" font-size="{font_px}" fill="{muted}">{escape(fmt(r[value_key]))}</text>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(parts)}</svg>'


SPECIMEN = {"agent": "#0a5d94", "control": "#b07400", "ink": "#15202b", "muted": "#4b5763",
            "grid": "#c2cbc4", "background": "#f8f9f6", "font": "'Public Sans','Helvetica Neue',Arial,sans-serif"}


def clade_around(tree: object, tip_name: str, min_tips: int = 20, max_tips: int = 45) -> object:
    """Smallest clade containing `tip_name` with at least `min_tips` tips (capped at `max_tips`)."""
    node = tree.find(tip_name)
    while node.parent is not None and node.count(tips=True) < min_tips:
        if node.parent.count(tips=True) > max_tips:
            break
        node = node.parent
    return node
