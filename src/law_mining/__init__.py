"""
Law mining module for ARC-AGI solver.

This module implements algorithmic law mining (M6) that sits on top of
the constraint-based kernel (M1-M5).

Key components:
  - roles.py: WL-style refinement to assign structural role_ids
  - role_stats.py: Aggregate role statistics from train/test examples
  - mine_s1_s2_s10.py: Miners for S1, S2, S10 schemas (M6.3A)
  - mine_s3_s4_s8_s9.py: Miners for S3, S4, S8, S9 schemas (M6.3B)
  - mine_s5_s6_s7_s11.py: Miners for S5, S6, S7, S11 schemas (M6.3C)
  - mine_law_config.py: Orchestrator that produces TaskLawConfig
"""

from src.law_mining.roles import compute_roles, RolesMapping, NodeKind
from src.law_mining.role_stats import compute_role_stats, RoleStats
from src.law_mining.mine_s1_s2_s10 import mine_S1, mine_S2, mine_S10
from src.law_mining.mine_s3_s4_s8_s9 import mine_S3, mine_S4, mine_S8, mine_S9
from src.law_mining.mine_s5_s6_s7_s11 import mine_S5, mine_S6, mine_S7, mine_S11

__all__ = [
    "compute_roles",
    "RolesMapping",
    "NodeKind",
    "compute_role_stats",
    "RoleStats",
    "mine_S1",
    "mine_S2",
    "mine_S10",
    "mine_S3",
    "mine_S4",
    "mine_S8",
    "mine_S9",
    "mine_S5",
    "mine_S6",
    "mine_S7",
    "mine_S11",
]
