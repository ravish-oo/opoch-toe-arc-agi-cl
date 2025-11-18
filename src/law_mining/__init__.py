"""
Law mining module for ARC-AGI solver.

This module implements algorithmic law mining (M6) that sits on top of
the constraint-based kernel (M1-M5).

Key components:
  - roles.py: WL-style refinement to assign structural role_ids
  - role_stats.py: Aggregate role statistics from train/test examples
  - mine_s*.py: Per-schema miners (S1-S11) that find always-true invariants
  - mine_law_config.py: Orchestrator that produces TaskLawConfig
"""

from src.law_mining.roles import compute_roles, RolesMapping, NodeKind
from src.law_mining.role_stats import compute_role_stats, RoleStats

__all__ = [
    "compute_roles",
    "RolesMapping",
    "NodeKind",
    "compute_role_stats",
    "RoleStats",
]
