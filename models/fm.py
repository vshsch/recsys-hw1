import torch
import numpy as np
from scipy.sparse import csr_matrix, hstack


class FM:
    def __init__(
        self, n_factors: int = 32, learning_rate: float = 1e-2,
        reg: float = 1e-5, n_epochs: int = 20, batch_size: int = 1024,
        seed: int = 42, device: str | None = None,
    ) -> None:
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")

    def fit(
        self, interactions: csr_matrix,
        user_features: csr_matrix, item_features: csr_matrix,
    ) -> "FM":
        self.user_features = csr_matrix(user_features, dtype=np.float32)
        self.item_features = csr_matrix(item_features, dtype=np.float32)
        self.user_dim = self.user_features.shape[1]
        n_features = self.user_dim + self.item_features.shape[1]
        rng = np.random.default_rng(self.seed)
        weights = torch.zeros(n_features,
                              device=self.device, requires_grad=True)
        factors = torch.tensor(
            0.05 * rng.standard_normal((n_features, self.n_factors)),
            dtype=torch.float32, device=self.device, requires_grad=True,
        )
        optimizer = torch.optim.Adam([weights, factors], lr=self.learning_rate)
        positives = interactions.tocoo()
        positive_users = positives.row
        positive_items = positives.col
        seen = interactions.toarray().astype(bool)
        self.history = []
        for epoch in range(1, self.n_epochs + 1):
            order = rng.permutation(len(positive_users))
            epoch_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = order[start:start + self.batch_size]
                users = positive_users[batch]
                items = positive_items[batch]
                negatives = rng.integers(interactions.shape[1],
                                         size=len(batch))
                collisions = seen[users, negatives]
                while collisions.any():
                    negatives[collisions] = rng.integers(
                        interactions.shape[1], size=collisions.sum(),
                    )
                    collisions = seen[users, negatives]
                positive_features = hstack([
                    self.user_features[users],
                    self.item_features[items],
                ], format="csr")
                negative_features = hstack([
                    self.user_features[users],
                    self.item_features[negatives],
                ], format="csr")
                positive_scores = self.forward(
                    self.to_torch(positive_features), weights, factors,
                )
                negative_scores = self.forward(
                    self.to_torch(negative_features), weights, factors,
                )
                loss = torch.nn.functional.softplus(
                    negative_scores - positive_scores,
                ).mean()
                loss += 0.5 * self.reg * (
                    weights.square().sum() + factors.square().sum()
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(batch)
            self.history.append({
                "epoch": epoch,
                "loss": epoch_loss / len(order),
            })
        self.weights = weights.detach().cpu().numpy()
        self.factors = factors.detach().cpu().numpy()
        self.prepare_scores(self.weights, self.factors)
        return self

    def to_torch(self, features: csr_matrix) -> torch.Tensor:
        features = features.tocoo()
        indices = torch.tensor(
            np.vstack([features.row, features.col]),
            dtype=torch.long, device=self.device,
        )
        values = torch.tensor(
            features.data, dtype=torch.float32, device=self.device,
        )
        return torch.sparse_coo_tensor(
            indices, values, features.shape, device=self.device,
            check_invariants=False,
        ).coalesce()

    def forward(
        self, features: torch.Tensor,
        weights: torch.Tensor, factors: torch.Tensor,
    ) -> torch.Tensor:
        linear = torch.sparse.mm(features, weights[:, None]).squeeze(1)
        projected = torch.sparse.mm(features, factors)
        squared_features = torch.sparse_coo_tensor(
            features.indices(), features.values().square(),
            features.shape, device=self.device, check_invariants=False,
        ).coalesce()
        squared = torch.sparse.mm(squared_features, factors.square())
        return linear + 0.5 * (projected.square() - squared).sum(dim=1)

    def prepare_scores(
        self, weights: np.ndarray, factors: np.ndarray,
    ) -> None:
        user_weights, item_weights = (
            weights[:self.user_dim], weights[self.user_dim:],
        )
        user_factors, item_factors = (
            factors[:self.user_dim], factors[self.user_dim:],
        )
        self.user_vectors = self.user_features @ user_factors
        self.item_vectors = self.item_features @ item_factors
        self.user_offsets = np.asarray(
            self.user_features @ user_weights,
        ).ravel() + 0.5 * np.sum(
            self.user_vectors ** 2
            - self.user_features.multiply(self.user_features)
            @ (user_factors ** 2),
            axis=1,
        )
        self.item_offsets = np.asarray(
            self.item_features @ item_weights,
        ).ravel() + 0.5 * np.sum(
            self.item_vectors ** 2
            - self.item_features.multiply(self.item_features)
            @ (item_factors ** 2),
            axis=1,
        )

    def scores(self, user_idx: int) -> np.ndarray:
        return (
            self.user_offsets[user_idx]
            + self.item_offsets
            + self.item_vectors @ self.user_vectors[user_idx]
        )
