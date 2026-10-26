import numpy as np
from scipy.sparse import csr_matrix


class CF:
    def __init__(
        self, n_neighbors: int = 100,
        shrinkage: float = 10.0, min_common: int = 2,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.shrinkage = shrinkage
        self.min_common = min_common

    def fit(self, interactions: csr_matrix) -> "CF":
        self.interactions = interactions
        sim = (interactions @ interactions.T).toarray().astype(np.float32)
        norms = np.sqrt(sim.diagonal().copy())
        sim[sim < self.min_common] = 0.0

        block_size = 1024
        for start in range(0, sim.shape[0], block_size):
            stop = min(start + block_size, sim.shape[0])
            block = sim[start:stop]
            if self.shrinkage > 0:
                block *= block / (block + self.shrinkage)
            block /= norms[start:stop, None]
            block /= norms[None, :]

        np.fill_diagonal(sim, 0.0)
        self.filter_neighbors(sim)
        self.similarity = csr_matrix(sim)
        return self

    def filter_neighbors(self, similarity: np.ndarray) -> None:
        if self.n_neighbors >= similarity.shape[1] - 1:
            return

        for row in similarity:
            if np.count_nonzero(row) <= self.n_neighbors:
                continue
            weakest = np.argpartition(
                row, -self.n_neighbors)[:-self.n_neighbors]
            row[weakest] = 0.0

    def scores(self, user_idx: int) -> np.ndarray:
        scores = self.similarity.getrow(user_idx) @ self.interactions
        return np.asarray(scores.toarray()).ravel()
