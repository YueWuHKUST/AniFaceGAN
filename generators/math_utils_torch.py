"""
Utilities for geometry etc.
"""

import torch


def transform_vectors(matrix: torch.Tensor, vectors4: torch.Tensor) -> torch.Tensor:
    """
    Left-multiplies MxM @ NxM. Returns NxM.
    """
    res = torch.matmul(vectors4, matrix.T)
    return res


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True))

def normalize_vecs_z(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths align z direction.
    """
    return vectors / torch.abs(vectors[...,-1:])

def torch_dot(x: torch.Tensor, y: torch.Tensor):
    """
    Dot product of two tensors.
    """
    return (x * y).sum(-1)
