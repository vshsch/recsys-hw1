import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


def interactions_to_csr(
    interactions: pl.DataFrame, user_col: str = "user_id", item_col: str = "item_id",
) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict[int, int], dict[int, int]]:
    user_ids = interactions[user_col].unique().sort().to_numpy()
    item_ids = interactions[item_col].unique().sort().to_numpy()

    user_to_index = {
        int(user_id): index for index, user_id in enumerate(user_ids)
    }
    item_to_index = {
        int(item_id): index for index, item_id in enumerate(item_ids)
    }

    rows = np.fromiter(
        (user_to_index[int(value)] for value in interactions[user_col]),
        dtype=np.int64,
        count=interactions.height,
    )
    columns = np.fromiter(
        (item_to_index[int(value)] for value in interactions[item_col]),
        dtype=np.int64,
        count=interactions.height,
    )

    matrix = csr_matrix(
        (np.ones(interactions.height, dtype=np.float32), (rows, columns)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data.fill(1.0)
    matrix.sort_indices()
    return matrix, user_ids, item_ids, user_to_index, item_to_index


def top_k_unseen(
    scores: np.ndarray, seen_indices: np.ndarray, k: int = 10,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or k <= 0:
        raise ValueError

    available = np.isfinite(scores)
    available[np.asarray(seen_indices, dtype=np.int64)] = False
    candidates = np.flatnonzero(available)
    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    k = min(k, candidates.size)
    candidate_scores = scores[candidates]
    if candidates.size > k:
        threshold = np.partition(candidate_scores, -k)[-k]
        above = candidates[candidate_scores > threshold]
        tied = candidates[candidate_scores == threshold]
        selected = np.concatenate([above, tied[:k - above.size]])
    else:
        selected = candidates

    order = np.lexsort((selected, -scores[selected]))
    return selected[order]


def recall_at_k(
    recommended: np.ndarray, relevant: set[int], k: int = 10,
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError
    return len(set(recommended[:k]) & relevant_set) / len(relevant_set)


def ndcg_at_k(
    recommended: np.ndarray, relevant: set[int], k: int = 10,
) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        raise ValueError
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, item_index in enumerate(recommended[:k])
        if item_index in relevant_set
    )
    ideal_len = min(k, len(relevant_set))
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_len))
    return float(dcg / idcg)
