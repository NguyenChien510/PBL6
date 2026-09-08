"""
Evaluation Metrics for Text-to-Video Person Retrieval (TVPR)
Calculates:
  - Cumulative Matching Characteristics (CMC): Rank@1, Rank@5, Rank@10, Rank@50
  - Median Rank (MdR) - System Stability Metric
  - Mean Reciprocal Rank (MRR)
  - Mean Average Precision (mAP)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def compute_similarity_matrix(query_feats: torch.Tensor, gallery_feats: torch.Tensor) -> np.ndarray:
    """
    Computes cosine similarity matrix between queries (Text) and gallery (Videos).
    Args:
        query_feats: Normalized text embeddings [N_q, D]
        gallery_feats: Normalized video embeddings [N_g, D]
    Returns:
        sim_matrix: Numpy array of shape [N_q, N_g]
    """
    if isinstance(query_feats, np.ndarray):
        query_feats = torch.from_numpy(query_feats)
    if isinstance(gallery_feats, np.ndarray):
        gallery_feats = torch.from_numpy(gallery_feats)

    # Normalize if not already unit vectors
    q = torch.nn.functional.normalize(query_feats, p=2, dim=-1)
    g = torch.nn.functional.normalize(gallery_feats, p=2, dim=-1)

    sim = torch.matmul(q, g.t()) # [N_q, N_g]
    return sim.cpu().numpy()


def evaluate_tvpr(
    sim_matrix: np.ndarray,
    query_labels: Union[np.ndarray, List[int]],
    gallery_labels: Union[np.ndarray, List[int]],
    ranks: List[int] = [1, 5, 10, 50]
) -> Dict[str, float]:
    """
    Evaluates retrieval performance.
    
    Args:
        sim_matrix: [N_q, N_g] similarity scores (higher = more similar)
        query_labels: Ground-truth IDs for each text query [N_q]
        gallery_labels: Ground-truth IDs for each video in gallery [N_g]
        ranks: List of rank thresholds to compute (default [1, 5, 10, 50])
    Returns:
        Dictionary containing Rank@1, Rank@5, Rank@10, Rank@50, MdR, MRR, mAP
    """
    query_labels = np.asarray(query_labels)
    gallery_labels = np.asarray(gallery_labels)
    num_queries, num_gallery = sim_matrix.shape

    first_ranks = []
    reciprocal_ranks = []
    aps = []

    for q_idx in range(num_queries):
        target_label = query_labels[q_idx]
        scores = sim_matrix[q_idx]

        # Sort gallery indices in descending order of similarity
        sorted_indices = np.argsort(-scores)
        sorted_labels = gallery_labels[sorted_indices]

        # Boolean mask indicating where gallery items match query
        matches = (sorted_labels == target_label)
        match_positions = np.where(matches)[0]

        if len(match_positions) > 0:
            # First match position (1-indexed rank)
            first_rank = match_positions[0] + 1
            first_ranks.append(first_rank)
            reciprocal_ranks.append(1.0 / first_rank)

            # Average Precision (AP) for this query
            cumulative_correct = np.cumsum(matches)
            rank_positions = np.arange(1, num_gallery + 1)
            precision_at_k = cumulative_correct / rank_positions
            ap = np.sum(precision_at_k * matches) / len(match_positions)
            aps.append(ap)
        else:
            first_ranks.append(num_gallery)
            reciprocal_ranks.append(0.0)
            aps.append(0.0)

    first_ranks = np.array(first_ranks)

    # Compute Rank@N accuracies
    results: Dict[str, float] = {}
    for r in ranks:
        r_accuracy = np.mean(first_ranks <= r) * 100.0
        results[f"Rank@{r}"] = float(round(r_accuracy, 2))

    # Median Rank (MdR) - Lower is better, indicates retrieval stability
    mdr = float(np.median(first_ranks))
    results["MdR"] = mdr

    # Mean Reciprocal Rank (MRR)
    mrr = float(round(np.mean(reciprocal_ranks) * 100.0, 2))
    results["MRR"] = mrr

    # Mean Average Precision (mAP)
    map_score = float(round(np.mean(aps) * 100.0, 2))
    results["mAP"] = map_score

    return results


def print_evaluation_results(metrics: Dict[str, float], title: str = "TVPR Evaluation"):
    """Prints pretty formatted evaluation metrics table."""
    print("=" * 60)
    print(f" {title.center(56)} ")
    print("=" * 60)
    for k, v in metrics.items():
        if k == "MdR":
            print(f"  {k:<15}: {v:.1f} (Median Rank - Stability Metric)")
        else:
            print(f"  {k:<15}: {v:.2f}%")
    print("=" * 60)
