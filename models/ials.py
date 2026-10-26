import numpy as np
from scipy.sparse import csr_matrix


class IALS:
    def __init__(
        self, n_factors: int = 16, reg: float = 1.0,
        alpha: float = 10.0, n_iters: int = 5, seed: int = 42,
    ) -> None:
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha
        self.n_iters = n_iters
        self.seed = seed

    def fit(self, interactions: csr_matrix) -> "IALS":
        n_users, n_items = interactions.shape
        rng = np.random.default_rng(self.seed)
        self.user_factors = np.zeros(
            (n_users, self.n_factors), dtype=np.float32,
        )
        self.item_factors = 0.05 * rng.standard_normal(
            (n_items, self.n_factors), dtype=np.float32,
        )
        by_item = interactions.T.tocsr()
        for _ in range(self.n_iters):
            self.user_factors = self.step(
                interactions, self.item_factors,
            )
            self.item_factors = self.step(
                by_item, self.user_factors,
            )
        return self

    def step(
        self, interactions_by_entity: csr_matrix, fixed_factors: np.ndarray,
    ) -> np.ndarray:
        factors = np.empty(
            (interactions_by_entity.shape[0],
             self.n_factors), dtype=np.float32,
        )
        base = fixed_factors.T @ fixed_factors + self.reg * np.eye(
            self.n_factors, dtype=np.float32,
        )
        for idx in range(interactions_by_entity.shape[0]):
            start, end = interactions_by_entity.indptr[idx:idx + 2]
            related_factors = fixed_factors[
                interactions_by_entity.indices[start:end]
            ]
            system = base + self.alpha * (
                related_factors.T @ related_factors
            )
            target = (1.0 + self.alpha) * related_factors.sum(axis=0)
            factors[idx] = np.linalg.solve(system, target)
        return factors

    def scores(self, user_idx: int) -> np.ndarray:
        return self.item_factors @ self.user_factors[user_idx]
