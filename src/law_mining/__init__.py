"""
Law mining module for ARC-AGI solver.

This module implements algorithmic law mining (M6) that sits on top of
the constraint-based kernel (M1-M5).

Key components:
  - roles.py: WL-style refinement to assign structural role_ids
  - role_stats.py: Aggregate role statistics from train/test examples
  - mine_s1_s2_s10.py: Miners for S1, S2, S10 schemas (M6.3A)
  - mine_law_config.py: Orchestrator that produces TaskLawConfig
"""

from src.law_mining.roles import compute_roles, RolesMapping, NodeKind
from src.law_mining.role_stats import compute_role_stats, RoleStats
from src.law_mining.mine_s1_s2_s10 import mine_S1, mine_S2, mine_S10

__all__ = [
    "compute_roles",
    "RolesMapping",
    "NodeKind",
    "compute_role_stats",
    "RoleStats",
    "mine_S1",
    "mine_S2",
    "mine_S10",
]
