import torch
import numpy as np


class TSVDPP:
    def __init__(
        self, n_users: int, n_items: int, n_factors: int = 32,
        learning_rate: float = 5e-3, reg: float = 2e-2,
        n_epochs: int = 20, n_time_bins: int = 30,
        seed: int = 42, device: str | None = None,
    ) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
        self.n_time_bins = n_time_bins
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")

    def fit(
        self, user_indices: np.ndarray, item_indices: np.ndarray,
        ratings: np.ndarray, timestamps: np.ndarray,
        user_histories: list[np.ndarray],
    ) -> "TSVDPP":
        rng = np.random.default_rng(self.seed)
        self.global_mean = float(np.mean(ratings))
        self.user_histories = [
            np.asarray(items, dtype=np.int64) for items in user_histories
        ]
        time_days = np.asarray(timestamps).astype(
            "datetime64[D]").astype(np.int64)
        self.min_time = int(time_days.min())
        self.max_time = int(time_days.max())
        time_span = self.max_time - self.min_time + 1
        time_bins = np.minimum(
            (time_days - self.min_time) * self.n_time_bins // time_span,
            self.n_time_bins - 1,
        )
        user_time_sum = np.zeros(self.n_users, dtype=np.float64)
        user_time_count = np.zeros(self.n_users, dtype=np.int64)
        np.add.at(user_time_sum, user_indices, time_days)
        np.add.at(user_time_count, user_indices, 1)
        self.user_mean_time = np.full(
            self.n_users, time_days.mean(), dtype=np.float64,
        )
        rated_users = user_time_count > 0
        self.user_mean_time[rated_users] = (
            user_time_sum[rated_users] / user_time_count[rated_users]
        )
        time_difference = time_days - self.user_mean_time[user_indices]
        time_deviation = (
            np.sign(time_difference) * np.abs(time_difference) ** 0.4
        ).astype(np.float32)
        rated_items = np.unique(item_indices)
        user_indices = torch.tensor(
            user_indices, dtype=torch.long, device=self.device,
        )
        item_indices = torch.tensor(
            item_indices, dtype=torch.long, device=self.device,
        )
        ratings = torch.tensor(
            ratings, dtype=torch.float32, device=self.device,
        )
        time_bins = torch.tensor(
            time_bins, dtype=torch.long, device=self.device,
        )
        time_deviation = torch.tensor(
            time_deviation, dtype=torch.float32, device=self.device,
        )
        history_matrix = self.history_matrix()
        user_bias = torch.zeros(
            self.n_users, device=self.device, requires_grad=True,
        )
        item_bias = torch.zeros(
            self.n_items, device=self.device, requires_grad=True,
        )
        user_bias_drift = torch.zeros(
            self.n_users, device=self.device, requires_grad=True,
        )
        item_time_bias = torch.zeros(
            (self.n_items, self.n_time_bins),
            device=self.device, requires_grad=True,
        )
        user_factors = torch.zeros(
            (self.n_users, self.n_factors),
            device=self.device, requires_grad=True,
        )
        user_factor_drift = torch.zeros(
            (self.n_users, self.n_factors),
            device=self.device, requires_grad=True,
        )
        initial_item_factors = np.zeros(
            (self.n_items, self.n_factors), dtype=np.float32,
        )
        initial_item_factors[rated_items] = 0.05 * rng.standard_normal(
            (len(rated_items), self.n_factors),
        )
        item_factors = torch.tensor(
            initial_item_factors,
            dtype=torch.float32, device=self.device, requires_grad=True,
        )
        history_factors = torch.zeros(
            (self.n_items, self.n_factors),
            device=self.device, requires_grad=True,
        )
        parameters = [
            user_bias, item_bias, user_bias_drift, item_time_bias,
            user_factors, user_factor_drift, item_factors, history_factors,
        ]
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        self.history = []
        for epoch in range(1, self.n_epochs + 1):
            implicit = torch.sparse.mm(history_matrix, history_factors)
            effective_user = (
                user_factors[user_indices]
                + user_factor_drift[user_indices] * time_deviation[:, None]
                + implicit[user_indices]
            )
            prediction = (
                self.global_mean
                + user_bias[user_indices]
                + user_bias_drift[user_indices] * time_deviation
                + item_bias[item_indices]
                + item_time_bias[item_indices, time_bins]
                + (item_factors[item_indices] * effective_user).sum(dim=1)
            )
            squared_error = (prediction - ratings).square().mean()
            penalty = (
                user_bias[user_indices].square().mean()
                + item_bias[item_indices].square().mean()
                + user_bias_drift[user_indices].square().mean()
                + item_time_bias[item_indices, time_bins].square().mean()
                + user_factors[user_indices].square().mean()
                + user_factor_drift[user_indices].square().mean()
                + item_factors[item_indices].square().mean()
                + history_factors.square().mean()
            )
            loss = squared_error + self.reg * penalty
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.history.append({
                "epoch": epoch,
                "train_rmse": float(torch.sqrt(squared_error).detach().cpu()),
            })
        with torch.no_grad():
            implicit = torch.sparse.mm(history_matrix, history_factors)
            self.user_bias = user_bias.detach().cpu().numpy()
            self.item_bias = item_bias.detach().cpu().numpy()
            self.user_bias_drift = user_bias_drift.detach().cpu().numpy()
            self.item_time_bias = item_time_bias.detach().cpu().numpy()
            self.user_factors = user_factors.detach().cpu().numpy()
            self.user_factor_drift = user_factor_drift.detach().cpu().numpy()
            self.item_factors = item_factors.detach().cpu().numpy()
            self.implicit_factors = implicit.detach().cpu().numpy()
        return self

    def history_matrix(self) -> torch.Tensor:
        counts = np.array(
            [len(items) for items in self.user_histories], dtype=np.int64,
        )
        rows = np.repeat(np.arange(self.n_users), counts)
        columns = np.concatenate(self.user_histories)
        values = np.repeat(
            np.divide(
                1.0, np.sqrt(counts),
                out=np.zeros(self.n_users), where=counts > 0,
            ),
            counts,
        )
        indices = torch.tensor(
            np.vstack([rows, columns]), dtype=torch.long, device=self.device,
        )
        values = torch.tensor(
            values, dtype=torch.float32, device=self.device,
        )
        return torch.sparse_coo_tensor(
            indices, values, (self.n_users, self.n_items), device=self.device,
            check_invariants=False,
        ).coalesce()

    def predict(
        self, user_idx: int, item_idx: int, timestamp: np.datetime64,
    ) -> float:
        return float(self.scores(user_idx, timestamp, np.array([item_idx]))[0])

    def scores(
        self, user_idx: int, timestamp: np.datetime64,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        if candidate_items is None:
            candidate_items = np.arange(self.n_items)
        time = int(np.datetime64(timestamp, "D").astype(np.int64))
        time_span = self.max_time - self.min_time + 1
        time_bin = np.clip(
            (time - self.min_time) * self.n_time_bins // time_span,
            0, self.n_time_bins - 1,
        )
        difference = time - self.user_mean_time[user_idx]
        deviation = np.sign(difference) * np.abs(difference) ** 0.4
        effective_user = (
            self.user_factors[user_idx]
            + self.user_factor_drift[user_idx] * deviation
            + self.implicit_factors[user_idx]
        )
        return (
            self.global_mean
            + self.user_bias[user_idx]
            + self.user_bias_drift[user_idx] * deviation
            + self.item_bias[candidate_items]
            + self.item_time_bias[candidate_items, time_bin]
            + self.item_factors[candidate_items] @ effective_user
        )
