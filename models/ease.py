import torch
import numpy as np
from scipy.sparse import csr_matrix


class EASE:
    def __init__(self, reg: float = 500.0, device: str | None = None) -> None:
        self.reg = reg
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")

    def fit(self, interactions: csr_matrix) -> "EASE":
        self.interactions = interactions
        gram = (interactions.T @ interactions).toarray().astype(
            np.float32, copy=False,
        )
        gram.flat[::gram.shape[0] + 1] += self.reg
        gram = torch.from_numpy(gram).to(self.device)
        precision = torch.cholesky_inverse(torch.linalg.cholesky(gram))
        weights = -precision / torch.diagonal(precision).unsqueeze(0)
        weights.fill_diagonal_(0.0)
        self.weights = weights.cpu().numpy()
        return self

    def scores(self, user_idx: int) -> np.ndarray:
        seen = self.interactions.getrow(user_idx).indices
        return self.weights[seen].sum(axis=0)
