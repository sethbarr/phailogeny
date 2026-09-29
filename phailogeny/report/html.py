"""Static HTML report generation for phailogeny analyses."""

from __future__ import annotations


def _escape_html(value: str) -> str:
    """Escape content so report HTML stays valid."""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def render_html_report(title: str, sections: dict[str, str]) -> str:
    """Render a static report with headings and readable content blocks."""
    body = []
    for heading, content in sections.items():
        safe_heading = _escape_html(str(heading))
        safe_content = _escape_html(str(content)).replace("\n", "<br>")
        body.append(f"<section><h2>{safe_heading}</h2><p>{safe_content}</p></section>")
    return f"<html><head><meta charset=\"utf-8\"><title>{_escape_html(str(title))}</title></head><body>{''.join(body)}</body></html>"
