from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class NoiseBatch:
    x_t: torch.Tensor
    edge_t: torch.Tensor
    active_t: torch.Tensor
    t: torch.Tensor
    p_t: torch.Tensor


class DiscreteGraphNoise:
    """DiGress-style categorical corruption for edges plus feature mixing.

    The molecule state uses rich supernode summaries, so node corruption mixes
    continuous summary features toward an empirical marginal while edges are
    sampled categorically from the bond marginal.
    """

    def __init__(
        self,
        timesteps: int,
        node_marginal: torch.Tensor,
        edge_marginal: torch.Tensor,
        node_noise_scale: float = 0.01,
        categorical_node_indices: tuple[int, ...] = (),
        reserved_node_indices: tuple[int, ...] = (),
        activity_marginal: float = 0.5,
        atom_feature_dim: int = 0,
    ) -> None:
        self.timesteps = int(timesteps)
        self.node_marginal = node_marginal.float()
        self.edge_marginal = edge_marginal.float() / edge_marginal.float().sum().clamp_min(1e-8)
        self.node_noise_scale = float(node_noise_scale)
        self.categorical_node_indices = tuple(int(value) for value in categorical_node_indices)
        self.reserved_node_indices = tuple(int(value) for value in reserved_node_indices)
        self.activity_marginal = float(min(1.0 - 1e-6, max(1e-6, activity_marginal)))
        self.atom_feature_dim = int(atom_feature_dim)

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.timesteps + 1, (batch_size,), device=device)

    def corruption_probability(self, t: torch.Tensor) -> torch.Tensor:
        return (t.float() / float(self.timesteps)).clamp(0.0, 1.0)

    def cumulative_retention(self, t: torch.Tensor | int) -> torch.Tensor:
        values = torch.as_tensor(t, dtype=torch.float32)
        return (1.0 - values / float(self.timesteps)).clamp(0.0, 1.0)

    @staticmethod
    def _categorical_matrix(retention: torch.Tensor, marginal: torch.Tensor) -> torch.Tensor:
        """Return rows ``q(x_t | x_0)`` for marginal replacement noise."""

        classes = int(marginal.numel())
        eye = torch.eye(classes, dtype=marginal.dtype, device=marginal.device)
        return retention * eye + (1.0 - retention) * marginal.view(1, classes).expand(classes, -1)

    def _categorical_reverse_probabilities(
        self,
        clean_probabilities: torch.Tensor,
        current_ids: torch.Tensor,
        t: torch.Tensor | int,
        s: torch.Tensor | int,
        marginal: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the exact ``q(x_s | x_t, x_0)`` posterior over predicted x0."""

        device = clean_probabilities.device
        dtype = clean_probabilities.dtype
        marginal = marginal.to(device=device, dtype=dtype)
        marginal = marginal / marginal.sum().clamp_min(1e-8)
        alpha_t = self.cumulative_retention(t).to(device=device, dtype=dtype)
        alpha_s = self.cumulative_retention(s).to(device=device, dtype=dtype)
        if alpha_t.numel() != 1 or alpha_s.numel() != 1:
            raise ValueError("Reverse transitions currently require scalar t and s values.")
        alpha_t = alpha_t.reshape(())
        alpha_s = alpha_s.reshape(())
        interval = torch.where(alpha_s > 0, alpha_t / alpha_s.clamp_min(1e-8), torch.zeros_like(alpha_t))
        qbar_t = self._categorical_matrix(alpha_t, marginal)
        qbar_s = self._categorical_matrix(alpha_s, marginal)
        q_t_given_s = self._categorical_matrix(interval, marginal)

        current = current_ids.long().clamp(min=0, max=marginal.numel() - 1)
        # qbar_t[x0, xt] and q_{t|s}[xs, xt] for every batch/pair position.
        denominator = qbar_t[:, current].movedim(0, -1).clamp_min(1e-12)
        transition_to_current = q_t_given_s[:, current].movedim(0, -1)
        weighted_clean = clean_probabilities / denominator
        previous = torch.matmul(weighted_clean, qbar_s)
        posterior = previous * transition_to_current
        return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _sample_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
        flat = probabilities.reshape(-1, probabilities.size(-1))
        return torch.multinomial(flat, 1).reshape(probabilities.shape[:-1])

    def corrupt(
        self,
        x: torch.Tensor,
        edge: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor | None = None,
        target_levels: torch.Tensor | None = None,
    ) -> NoiseBatch:
        device = x.device
        batch_size, num_nodes, _ = x.shape
        if t is None:
            t = self.sample_timesteps(batch_size, device=device)
        p = self.corruption_probability(t).view(batch_size, 1, 1)

        marginal = self.node_marginal.to(device=device, dtype=x.dtype).view(1, 1, -1)
        eps = torch.randn_like(x) * self.node_noise_scale
        x_t = (1.0 - p) * x + p * marginal + eps * p

        if self.atom_feature_dim > 0 and target_levels is not None:
            atom_rows = target_levels.long().view(-1) == 0
            if atom_rows.any():
                atom_marginal = self.node_marginal[: self.atom_feature_dim].to(device=device).clamp_min(0.0)
                atom_marginal = atom_marginal / atom_marginal.sum().clamp_min(1e-8)
                replacement = torch.multinomial(
                    atom_marginal,
                    int(atom_rows.sum().item()) * num_nodes,
                    replacement=True,
                ).view(-1, num_nodes)
                clean_ids = x[atom_rows, :, : self.atom_feature_dim].argmax(dim=-1)
                retain = torch.rand(int(atom_rows.sum().item()), num_nodes, device=device) >= p[atom_rows].squeeze(-1)
                atom_ids = torch.where(retain, clean_ids, replacement)
                x_t[atom_rows, :, : self.atom_feature_dim] = torch.nn.functional.one_hot(
                    atom_ids, num_classes=self.atom_feature_dim
                ).to(dtype=x.dtype)

        for index in self.categorical_node_indices:
            probability = marginal[..., index].clamp(0.0, 1.0)
            replacement = torch.rand(batch_size, num_nodes, device=device) < probability
            retain = torch.rand(batch_size, num_nodes, device=device) >= p.squeeze(-1)
            x_t[..., index] = torch.where(retain, x[..., index] > 0.5, replacement).to(dtype=x.dtype)
        for index in self.reserved_node_indices:
            x_t[..., index] = 0.0

        active_replacement = torch.rand(batch_size, num_nodes, device=device) < self.activity_marginal
        active_retain = torch.rand(batch_size, num_nodes, device=device) >= p.squeeze(-1)
        active_t = torch.where(active_retain, mask, active_replacement)
        x_t = x_t * active_t.unsqueeze(-1).to(dtype=x.dtype)

        edge_marginal = self.edge_marginal.to(device=device)
        random_edges = torch.multinomial(edge_marginal, batch_size * num_nodes * num_nodes, replacement=True)
        random_edges = random_edges.view(batch_size, num_nodes, num_nodes)
        replace = torch.rand(batch_size, num_nodes, num_nodes, device=device) < p
        edge_t = torch.where(replace, random_edges, edge)
        edge_t = torch.triu(edge_t, diagonal=1)
        edge_t = edge_t + edge_t.transpose(1, 2)
        diag = torch.arange(num_nodes, device=device)
        edge_t[:, diag, diag] = 0
        valid_pair = active_t.unsqueeze(1) & active_t.unsqueeze(2)
        edge_t = torch.where(valid_pair, edge_t, torch.zeros_like(edge_t))
        return NoiseBatch(x_t=x_t, edge_t=edge_t.long(), active_t=active_t, t=t, p_t=p.view(batch_size))

    def reverse_step(
        self,
        *,
        clean_x: torch.Tensor,
        clean_edge_logits: torch.Tensor,
        clean_active_logits: torch.Tensor,
        current_x: torch.Tensor,
        current_edge: torch.Tensor,
        current_active: torch.Tensor,
        t: int,
        s: int,
        none_idx: int = 0,
        edge_temperature: float = 1.0,
        none_bias: float = 0.0,
        max_degree: int = 0,
        categorical_atom: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a skipped reverse transition from diffusion time ``t`` to ``s``.

        Each categorical field uses its marginal-replacement posterior under
        the predicted clean-state distribution.  Edge and activity variables
        are sampled factorwise and then projected jointly so edges incident to
        inactive nodes become no-edge.  Numerical summaries use the analogous
        predicted-clean re-noising update.
        """

        if not (0 <= int(s) < int(t) <= self.timesteps):
            raise ValueError(f"Expected 0 <= s < t <= T, got s={s}, t={t}, T={self.timesteps}.")
        device = clean_x.device
        p_s = float(s) / float(self.timesteps)
        marginal_x = self.node_marginal.to(device=device, dtype=clean_x.dtype).view(1, 1, -1)
        eps = torch.randn_like(clean_x) * self.node_noise_scale
        x_s = (1.0 - p_s) * clean_x + p_s * marginal_x + p_s * eps

        if categorical_atom and self.atom_feature_dim > 0:
            atom_marginal = self.node_marginal[: self.atom_feature_dim].to(device=device, dtype=clean_x.dtype)
            atom_marginal = atom_marginal.clamp_min(0.0)
            atom_marginal = atom_marginal / atom_marginal.sum().clamp_min(1e-8)
            clean_atom_probability = torch.softmax(clean_x[..., : self.atom_feature_dim], dim=-1)
            current_atom = current_x[..., : self.atom_feature_dim].argmax(dim=-1)
            atom_posterior = self._categorical_reverse_probabilities(
                clean_atom_probability,
                current_atom,
                t=t,
                s=s,
                marginal=atom_marginal,
            )
            atom_ids = self._sample_probabilities(atom_posterior)
            x_s[..., : self.atom_feature_dim] = torch.nn.functional.one_hot(
                atom_ids, num_classes=self.atom_feature_dim
            ).to(dtype=clean_x.dtype)

        for index in self.categorical_node_indices:
            probability_one = float(self.node_marginal[index].clamp(1e-6, 1.0 - 1e-6))
            binary_marginal = torch.tensor(
                [1.0 - probability_one, probability_one], device=device, dtype=clean_x.dtype
            )
            probability_one_clean = torch.sigmoid(clean_x[..., index])
            clean_probability = torch.stack([1.0 - probability_one_clean, probability_one_clean], dim=-1)
            current = (current_x[..., index] > 0.5).long()
            posterior = self._categorical_reverse_probabilities(
                clean_probability, current, t=t, s=s, marginal=binary_marginal
            )
            x_s[..., index] = self._sample_probabilities(posterior).to(dtype=clean_x.dtype)
        for index in self.reserved_node_indices:
            x_s[..., index] = 0.0

        clean_active_one = torch.sigmoid(clean_active_logits)
        clean_active_probability = torch.stack([1.0 - clean_active_one, clean_active_one], dim=-1)
        active_marginal = torch.tensor(
            [1.0 - self.activity_marginal, self.activity_marginal],
            device=device,
            dtype=clean_x.dtype,
        )
        active_posterior = self._categorical_reverse_probabilities(
            clean_active_probability,
            current_active.long(),
            t=t,
            s=s,
            marginal=active_marginal,
        )
        active_s = self._sample_probabilities(active_posterior).bool()
        empty = ~active_s.any(dim=1)
        if empty.any():
            best = clean_active_logits[empty].argmax(dim=1)
            active_s[empty, best] = True

        logits = clean_edge_logits / max(float(edge_temperature), 1e-6)
        logits = logits.clone()
        logits[..., int(none_idx)] += float(none_bias)
        clean_edge_probability = torch.softmax(logits, dim=-1)
        edge_posterior = self._categorical_reverse_probabilities(
            clean_edge_probability,
            current_edge.long(),
            t=t,
            s=s,
            marginal=self.edge_marginal,
        )
        edge_s = self._sample_probabilities(edge_posterior)
        edge_s = torch.triu(edge_s, diagonal=1)
        edge_s = edge_s + edge_s.transpose(1, 2)
        diag = torch.arange(edge_s.size(1), device=device)
        edge_s[:, diag, diag] = int(none_idx)
        valid_pair = active_s.unsqueeze(1) & active_s.unsqueeze(2)
        edge_s = torch.where(valid_pair, edge_s, torch.full_like(edge_s, int(none_idx)))

        if max_degree > 0:
            # Deterministically remove the least probable sampled non-edge
            # candidates first, preserving symmetry.
            non_none_score = 1.0 - clean_edge_probability[..., int(none_idx)]
            for batch_index in range(edge_s.size(0)):
                active = active_s[batch_index].nonzero(as_tuple=False).flatten().tolist()
                while True:
                    degrees = {
                        node: sum(
                            int(edge_s[batch_index, node, neighbor].item()) != int(none_idx)
                            for neighbor in active
                            if neighbor != node
                        )
                        for node in active
                    }
                    overflowing = [node for node, degree in degrees.items() if degree > max_degree]
                    if not overflowing:
                        break
                    node = max(overflowing, key=lambda value: degrees[value])
                    incident = [
                        neighbor
                        for neighbor in active
                        if neighbor != node and int(edge_s[batch_index, node, neighbor].item()) != int(none_idx)
                    ]
                    drop = min(incident, key=lambda value: float(non_none_score[batch_index, node, value]))
                    edge_s[batch_index, node, drop] = int(none_idx)
                    edge_s[batch_index, drop, node] = int(none_idx)

        x_s = x_s * active_s.unsqueeze(-1).to(dtype=x_s.dtype)
        return x_s, edge_s.long(), active_s


def estimate_marginals(dataset, node_feature_dim: int, edge_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    node_sum = torch.zeros(node_feature_dim, dtype=torch.float32)
    node_count = 0.0
    edge_counts = torch.ones(edge_dim, dtype=torch.float32)
    for idx in range(len(dataset)):
        item = dataset[idx]
        target = item["target"]
        mask = target["mask"]
        x = target["x"]
        edge = target["edge"]
        if mask.any():
            node_sum += x[mask].sum(dim=0)
            node_count += float(mask.sum().item())
        valid = torch.triu(mask.unsqueeze(0) & mask.unsqueeze(1), diagonal=1)
        values = edge[valid]
        if values.numel() > 0:
            edge_counts += torch.bincount(values.long(), minlength=edge_dim).float()
    node_marginal = node_sum / max(1.0, node_count)
    edge_marginal = edge_counts / edge_counts.sum().clamp_min(1.0)
    return node_marginal, edge_marginal


def estimate_activity_marginal(dataset) -> float:
    active = 0.0
    capacity = 0.0
    for idx in range(len(dataset)):
        mask = dataset[idx]["target"]["mask"]
        active += float(mask.sum().item())
        capacity += float(mask.numel())
    return active / max(1.0, capacity)
