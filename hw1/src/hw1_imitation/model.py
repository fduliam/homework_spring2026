"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        self.activation = nn.ReLU()
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk = action_chunk.view(-1, self.chunk_size * self.action_dim)
        action_chunk_pred = self.forward(state)
        action_chunk_pred = action_chunk_pred.view(-1, self.chunk_size * self.action_dim)
        loss = torch.mean((action_chunk - action_chunk_pred) ** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_chunk = self.forward(state)
        return action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim + action_dim * chunk_size + 1, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        self.layers.append(nn.Linear(hidden_dims[-1], action_dim * chunk_size))
        self.activation = nn.ReLU()

    def forward(self, state: torch.Tensor, action_chunk: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action_chunk, t], dim=-1)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_chunk_noisy =  torch.randn_like(action_chunk)
        action_chunk_noisy = action_chunk_noisy.view(-1, self.chunk_size * self.action_dim)
        batch_size = state.shape[0]
        t = torch.rand(batch_size, 1, device=state.device)
        action_chunk = action_chunk.view(-1, self.chunk_size * self.action_dim)
        velocity_pred = self.forward(state, action_chunk_noisy + t * (action_chunk - action_chunk_noisy), t)
        velocity_pred = velocity_pred.view(-1, self.chunk_size * self.action_dim)
        loss = torch.mean((velocity_pred - (action_chunk - action_chunk_noisy))** 2)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        x = torch.randn(batch_size, self.chunk_size * self.action_dim, device=state.device)  # x_0
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.full((batch_size, 1), step * dt, device=state.device)
            velocity = self.forward(state, x, t)
            velocity = velocity.view(-1, self.chunk_size * self.action_dim)
            x = x + dt * velocity
        return x.view(-1, self.chunk_size, self.action_dim)


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
