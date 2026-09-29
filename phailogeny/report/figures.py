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


def clade_around(tree: object, tip_name: str, min_tips: int = 20, max_tips: int = 45) -> object:
    """Smallest clade containing `tip_name` with at least `min_tips` tips (capped at `max_tips`)."""
    node = tree.find(tip_name)
    while node.parent is not None and node.count(tips=True) < min_tips:
        if node.parent.count(tips=True) > max_tips:
            break
        node = node.parent
    return node
