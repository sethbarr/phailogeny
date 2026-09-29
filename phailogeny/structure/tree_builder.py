"""Tree-building helpers based on the equal-weight distance matrix."""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform


def _nearest_neighbors(matrix: dict[str, list[float]], limit: int = 5) -> list[dict[str, float | str]]:
    """Return the closest pairwise matches from the similarity matrix."""
    names = list(matrix)
    pairs: list[dict[str, float | str]] = []
    for index, left in enumerate(names):
        for right_index in range(index + 1, len(names)):
            right = names[right_index]
            distance = float(matrix[left][right_index])
            pairs.append({"left": left, "right": right, "distance": distance})
    pairs.sort(key=lambda item: float(item["distance"]))
    return pairs[:limit]


def _newick_from_linkage(Z: np.ndarray, labels: list[str]) -> str:
    """Convert a SciPy linkage matrix into a Newick-form tree string."""
    tree = to_tree(Z, False)

    def walk(node: object) -> str:
        if hasattr(node, "left") and node.left is not None and hasattr(node, "right") and node.right is not None:
            left = walk(node.left)
            right = walk(node.right)
            branch_length = float(node.dist) if getattr(node, "dist", 0.0) is not None else 0.0
            left_length = max(branch_length / 2.0, 0.0)
            right_length = max(branch_length / 2.0, 0.0)
            return f"({left}:{left_length},{right}:{right_length})"
        leaf_id = int(node.id)
        return labels[leaf_id]

    return f"{walk(tree)};"


def nj_tree_from_matrix(matrix: dict[str, list[float]]) -> dict[str, object]:
    """Build a real distance-based hierarchy from the matrix.

    This uses agglomerative clustering on the pairwise distance matrix so the tree
    reflects actual similarity structure instead of a flat list of leaves.
    """
    if not matrix:
        return {"nodes": [], "newick": "()", "nearest_neighbors": []}

    names = list(matrix)
    if len(names) == 1:
        return {"nodes": names, "newick": f"({names[0]});", "nearest_neighbors": []}

    array = np.asarray([[float(matrix[name][index]) for index, _ in enumerate(names)] for name in names], dtype=float)
    condensed = squareform(array, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    newick = _newick_from_linkage(linkage_matrix, names)
    return {
        "nodes": [str(name) for name in names],
        "newick": newick,
        "nearest_neighbors": _nearest_neighbors(matrix),
    }
