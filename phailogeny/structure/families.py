"""Module-family grouping utilities."""

from __future__ import annotations


def group_module_families(modules: list[dict[str, object]], threshold: float = 0.8) -> dict[str, list[str]]:
    """Group modules by a simple shared-content heuristic to mirror the eventual MinHash family search."""
    families: dict[str, list[str]] = {}
    for module in modules:
        module_id = str(module.get("module_id") or module.get("label") or "unknown")
        family_key = str(module.get("content") or "").lower().strip()
        if not family_key:
            family_key = "empty"
        if family_key not in families:
            families[family_key] = []
        families[family_key].append(module_id)
    return {key: value for key, value in families.items() if len(value) > 0 and threshold <= 1.0}
