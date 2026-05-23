from __future__ import annotations

from time import perf_counter
from typing import Dict

import torch

from evals.simu_env_planning.planning.planning.csa.support_memory import SupportMemory


class ConditionalSupportScorer:
    """Chunked PyTorch implementation of conditional state-action support."""

    def __init__(
        self,
        memory: SupportMemory,
        state_k: int = 64,
        action_k: int = 64,
        beta: float = 1.0,
        chunk_size: int = 8192,
    ) -> None:
        if state_k <= 0:
            raise ValueError(f"state_k must be positive, got {state_k}")
        if action_k <= 0:
            raise ValueError(f"action_k must be positive, got {action_k}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.memory = memory
        self.state_k = int(state_k)
        self.action_k = int(action_k)
        self.beta = float(beta)
        self.chunk_size = int(chunk_size)

    def score(
        self,
        states,
        actions,
        raw_state: bool = True,
        state_reduction: str | None = None,
    ) -> Dict[str, torch.Tensor | float]:
        start = perf_counter()
        query_states = self.memory.transform_query_state(states, raw_state=raw_state, state_reduction=state_reduction)
        query_actions = self.memory.normalize_query_actions(actions)
        query_states = query_states.to(device=self.memory.state_features.device, dtype=self.memory.state_features.dtype)
        query_actions = query_actions.to(device=self.memory.actions.device, dtype=self.memory.actions.dtype)

        state_score, state_nn_idx = self._state_scores(query_states)
        action_score = self._action_scores(query_actions, state_nn_idx)
        csa_score = state_score + self.beta * action_score
        return {
            "state_score": state_score,
            "action_score": action_score,
            "csa_score": csa_score,
            "score_latency_ms": (perf_counter() - start) * 1000.0,
        }

    def _state_scores(self, query_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k = min(self.state_k, self.memory.num_items)
        best_dists = None
        best_idxs = None
        memory_states = self.memory.state_features.to(device=query_states.device, dtype=query_states.dtype)
        for start in range(0, self.memory.num_items, self.chunk_size):
            end = min(start + self.chunk_size, self.memory.num_items)
            dists = torch.cdist(query_states, memory_states[start:end])
            vals, idx = torch.topk(dists, k=min(k, dists.shape[1]), largest=False, dim=1)
            idx = idx + start
            if best_dists is None:
                best_dists, best_idxs = vals, idx
            else:
                merged_dists = torch.cat([best_dists, vals], dim=1)
                merged_idxs = torch.cat([best_idxs, idx], dim=1)
                vals, order = torch.topk(merged_dists, k=k, largest=False, dim=1)
                best_dists = vals
                best_idxs = torch.gather(merged_idxs, 1, order)
        return best_dists.mean(dim=1), best_idxs

    def _action_scores(self, query_actions: torch.Tensor, state_nn_idx: torch.Tensor) -> torch.Tensor:
        local_actions = self.memory.normalized_actions().to(device=query_actions.device, dtype=query_actions.dtype)
        local_actions = local_actions[state_nn_idx]
        action_dists = torch.linalg.vector_norm(local_actions - query_actions.unsqueeze(1), dim=-1)
        k = min(self.action_k, action_dists.shape[1])
        return torch.topk(action_dists, k=k, largest=False, dim=1).values.mean(dim=1)
