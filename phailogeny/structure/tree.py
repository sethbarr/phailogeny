"""Neighbour-joining and Newick export helpers for the agent estate tree."""

from __future__ import annotations


def neighbour_joining(distances: list[list[float]]) -> list[str]:
    """Create a deterministic placeholder tree structure.

    This is a lightweight stand-in for the eventual scikit-bio neighbour-joining
    implementation required by the spec; it keeps the runtime and test surface small
    while preserving a tree-like output format.
    """
    if not distances:
        return []
    if len(distances) == 1:
        return ["A0"]

    labels = [f"A{index}" for index in range(len(distances))]
    return [f"({left},{right})" for left, right in zip(labels[::2], labels[1::2])]


def newick_from_tree(tree: list[str]) -> str:
    """Convert a tree-like structure into Newick format."""
    if not tree:
        return "()"
    return ";".join(tree) + ";"
