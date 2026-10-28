from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MGPConfig:
    input_mode: str = "seurat"
    K: int = 50
    nu: float = 3.0
    a1: float = 2.1
    a2: float = 3.1
    a_sigma: float = 1.0
    b_sigma: float = 1.0
    precision: str = "float64"
    tol: float = 0.1
    drop_budget: float = 0.5
    max_iter: int = 1000
    n_iter: int = 20000
    burn_in: int = 10000
    thin: int = 10
    seed: int = 2026
    batch_size: int = 32
    n_columns: int = 20
    n_neighbors: int = 10
    leiden_resolution: float = 0.45
    random_state: int = 0
    representation: str = "d20_norm"

    def __post_init__(self):
        if self.input_mode != "seurat":
            raise ValueError("the fixed workflow requires input_mode='seurat'")
        if self.representation != "d20_norm":
            raise ValueError("the fixed workflow requires representation='d20_norm'")
        if self.K < 1 or self.thin < 1 or self.batch_size < 1:
            raise ValueError("K, thin and batch_size must be positive")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")
        if not 0 <= self.burn_in < self.n_iter:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_iter")
        if (self.n_iter - self.burn_in) // self.thin < 1:
            raise ValueError("the Gibbs run would retain no draws")
        if not 1 <= self.n_columns <= self.K:
            raise ValueError("n_columns must be in [1, K]")
        if self.tol <= 0 or self.drop_budget < 0 or self.max_iter < 1:
            raise ValueError("invalid CAVI stopping settings")

    @property
    def standardize(self) -> bool:
        return False

    @property
    def n_kept(self) -> int:
        return (self.n_iter - self.burn_in) // self.thin

    @property
    def input_tag(self) -> str:
        return "seurat"
