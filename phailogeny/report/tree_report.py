"""Static HTML report with an inline SVG rendering of the rooted NJ tree."""

from __future__ import annotations

from html import escape

PALETTE = ["#2a78d6", "#d9542b", "#1f9e6e", "#a05bd0", "#c9a227", "#d6457f", "#3aa6b8", "#7a7a7a"]
ROW = 14
LABEL_WIDTH = 360
TREE_WIDTH = 620


def tree_svg(tree: object, source_of: dict[str, str]) -> str:
    """Draw a rectangular phylogram. X = cumulative branch length, tips evenly spaced."""
    sources = sorted(set(source_of.values()))
    colour = {src: PALETTE[i % len(PALETTE)] for i, src in enumerate(sources)}

    depth: dict[int, float] = {}
    for node in tree.preorder():
        parent_depth = depth[id(node.parent)] if node.parent is not None else 0.0
        depth[id(node)] = parent_depth + max(float(node.length or 0.0), 0.0) * (node.parent is not None)
    max_depth = max(depth.values()) or 1.0
    scale = TREE_WIDTH / max_depth

    y: dict[int, float] = {}
    tips = list(tree.tips())
    for index, tip in enumerate(tips):
        y[id(tip)] = 20 + index * ROW
    for node in tree.postorder():
        if node.children:
            ys = [y[id(c)] for c in node.children]
            y[id(node)] = (min(ys) + max(ys)) / 2

    parts: list[str] = []
    for node in tree.preorder():
        x = 10 + depth[id(node)] * scale
        if node.children:
            ys = [y[id(c)] for c in node.children]
            parts.append(f'<line x1="{x:.1f}" y1="{min(ys):.1f}" x2="{x:.1f}" y2="{max(ys):.1f}" class="br"/>')
            for child in node.children:
                cx = 10 + depth[id(child)] * scale
                cy = y[id(child)]
                parts.append(f'<line x1="{x:.1f}" y1="{cy:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" class="br"/>')
        else:
            name = str(node.name).replace("_", " ") if node.name else ""
            src = source_of.get(str(node.name), "")
            parts.append(
                f'<text x="{x + 4:.1f}" y="{y[id(node)] + 4:.1f}" fill="{colour.get(src, "#555")}">'
                f"{escape(name)}</text>"
            )

    height = 40 + len(tips) * ROW
    width = TREE_WIDTH + LABEL_WIDTH + 20
    legend = " ".join(
        f'<span class="key"><i style="background:{colour[s]}"></i>{escape(s)}</span>' for s in sources
    )
    return (
        f'<div class="legend">{legend}</div><div class="scroll">'
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">{"".join(parts)}</svg></div>'
    )


def render_tree_report(result: dict[str, object], pairs: list[dict[str, object]], clusters: list[list[str]]) -> str:
    """Assemble the full HTML report."""
    source_of = {aid: str(rec["source"]) for aid, rec in zip(result["ids"], result["records"])}
    coverage = ", ".join(f"{k} {v:.0%}" for k, v in result["coverage"].items())
    weights = ", ".join(f"{k}={v:g}" for k, v in result["weights"].items())

    pair_rows = "".join(
        f"<tr><td>{escape(p['left'])}</td><td>{escape(p['right'])}</td>"
        f"<td>{p['functional_similarity']:.2f}</td><td>{p['lineage_similarity']:.2f}</td>"
        f"<td class=\"{p['relation']}\">{p['relation']}</td></tr>"
        for p in pairs
    )
    cluster_items = "".join(
        f"<li><b>{len(c)}</b> · {escape(', '.join(c))}</li>" for c in sorted(clusters, key=len, reverse=True)
    )
    copies = [(aid, rec["copies"]) for aid, rec in zip(result["ids"], result["records"]) if rec.get("copies")]
    copy_items = "".join(
        f"<li><b>{escape(aid)}</b> also at: {escape(', '.join(paths))}</li>" for aid, paths in copies
    ) or "<li>none</li>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>phailogeny tree</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#1b1b1b;background:#fafaf8}}
h1{{margin:0 0 4px}} .meta{{color:#666;margin-bottom:20px}}
section{{background:#fff;border:1px solid #e3e3e0;border-radius:10px;padding:16px;margin:16px 0}}
.scroll{{overflow:auto;max-height:80vh;border-top:1px solid #eee}}
svg text{{font:11px ui-monospace,monospace}} .br{{stroke:#444;stroke-width:1}}
.legend{{margin-bottom:8px}} .key{{margin-right:14px}}
.key i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} td,th{{padding:4px 8px;border-bottom:1px solid #eee;text-align:left}}
.homologous{{color:#b04a1c;font-weight:600}} .convergent{{color:#1f7a55;font-weight:600}}
</style></head><body>
<h1>Agent similarity tree</h1>
<div class="meta">{len(result['ids'])} agents · block coverage: {coverage} · weights: {weights}<br>
Neighbour-joining on combined functional distance (purpose + prompt embeddings, IDF-weighted tools), midpoint-rooted.
Lineage (MinHash text reuse) is kept separate and only used to label pairs.</div>
<section><h2>Closest pairs</h2>
<p><b>homologous</b> = high literal text overlap (copied / forked). <b>convergent</b> = similar job, different text.</p>
<table><tr><th>Agent</th><th>Agent</th><th>Functional sim</th><th>Lineage sim</th><th>Relation</th></tr>{pair_rows}</table></section>
<section><h2>Tree</h2>{tree_svg(result['tree'], source_of)}</section>
<section><h2>Clusters (HDBSCAN)</h2><ul>{cluster_items}</ul></section>
<section><h2>Exact copies (collapsed to one leaf)</h2><p>Identical purpose + prompt. Copied, not referenced: fixes won't propagate.</p><ul>{copy_items}</ul></section>
</body></html>"""
