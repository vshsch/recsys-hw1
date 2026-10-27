import time
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix


def interactions_to_csr(
    interactions: pl.DataFrame,
    user_col: str = "user_id", item_col: str = "item_id",
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


def evaluate(
    model: object, evaluation: pl.DataFrame, seen_matrix: csr_matrix,
    user_to_index: dict[int, int], item_to_index: dict[int, int],
    user_col: str = "user_id", item_col: str = "item_id", k: int = 10,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if evaluation[user_col].n_unique() != evaluation.height:
        raise ValueError

    rows = []
    for user_id, item_id in evaluation.select(user_col, item_col).iter_rows():
        user_id, item_id = int(user_id), int(item_id)
        if user_id not in user_to_index or item_id not in item_to_index:
            raise ValueError

        user_index = user_to_index[user_id]
        target_index = item_to_index[item_id]
        seen_indices = seen_matrix.getrow(user_index).indices
        if target_index in seen_indices:
            raise ValueError

        scores = np.asarray(model.scores(user_index), dtype=float)
        if scores.shape != (seen_matrix.shape[1],):
            raise ValueError

        recommended = top_k_unseen(scores, seen_indices, k=k)
        matches = np.flatnonzero(recommended == target_index)
        rank = int(matches[0] + 1) if matches.size else None
        relevant = {target_index}
        rows.append({
            user_col: user_id,
            "rank": rank,
            f"Recall@{k}": recall_at_k(recommended, relevant, k=k),
            f"NDCG@{k}": ndcg_at_k(recommended, relevant, k=k),
        })
    details = pl.DataFrame(rows, infer_schema_length=None)
    summary = details.select(
        pl.len().alias("users"),
        pl.col(f"Recall@{k}").mean(),
        pl.col(f"NDCG@{k}").mean(),
    )
    return summary, details


def run_grid_search(
    model_class: type, model_name: str, configs: list[dict], fit_args: tuple,
    evaluation: pl.DataFrame, seen_matrix: csr_matrix,
    user_to_index: dict[int, int], item_to_index: dict[int, int], k: int = 10,
) -> tuple[list[dict], object, dict, pl.DataFrame]:
    rows = []
    best_score = -np.inf
    for config_number, parameters in enumerate(configs, start=1):
        print(f"{model_name} {config_number}/{len(configs)}: {parameters}")

        started = time.perf_counter()
        model = model_class(**parameters).fit(*fit_args)
        fit_seconds = time.perf_counter() - started

        started = time.perf_counter()
        metrics, details = evaluate(
            model, evaluation, seen_matrix,
            user_to_index, item_to_index, k=k,
        )
        eval_seconds = time.perf_counter() - started

        row = {
            "model": model_name,
            "parameters": str(parameters),
            "fit_sec": round(fit_seconds, 3),
            "eval_sec": round(eval_seconds, 3),
            **metrics.row(0, named=True),
        }
        rows.append(row)
        if row[f"NDCG@{k}"] > best_score:
            best_score = row[f"NDCG@{k}"]
            best_model = model
            best_parameters = parameters.copy()
            best_details = details
        print(
            f"  NDCG@{k}={row[f'NDCG@{k}']:.4f}, "
            f"Recall@{k}={row[f'Recall@{k}']:.4f}"
        )
    return rows, best_model, best_parameters, best_details


class Popularity:
    def fit(self, interactions: csr_matrix) -> "Popularity":
        self.item_scores = np.asarray(interactions.sum(axis=0)).ravel()
        return self

    def scores(self, user_idx: int) -> np.ndarray:
        return self.item_scores
