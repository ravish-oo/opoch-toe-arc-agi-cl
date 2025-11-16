## Repo layout

```text
.
├── data/
│   ├── arc-agi_training_challenges.json
│   ├── arc-agi_training_solutions.json
│   ├── arc-agi_evaluation_challenges.json
│   ├── arc-agi_evaluation_solutions.json
│   ├── arc-agi_test_challenges.json
│   ├── arc1_training.json
│   ├── arc2_training.json
│   └── ... (as you have now)
│
├── docs/
│   ├── anchors/                    # math spec you pasted (φ, S1–S11, etc.)
│   ├── human_pair_programmer_docs/
│   └── implementation_plan.md
│
├── src/
│   ├── core/
│   │   ├── grid_types.py           # Grid type alias, Pixel, Component dataclass, basic helpers
│   │   └── arc_io.py               # load/save ARC tasks from data/*.json
│   │
│   ├── features/                   # φ(p) operators
│   │   ├── coords_bands.py         # coord_features, row/col bands, border_mask
│   │   ├── components.py           # connected_components_by_color, shape_signature, object_id
│   │   └── neighborhoods.py        # row/col flags, neighborhood_hashes, etc.
│   │
│   ├── constraints/
│   │   ├── indexing.py             # pixel/color → y-index mapping (idx(p,c), index_to_pixel)
│   │   ├── builder.py              # LinearConstraint, ConstraintBuilder, one-hot constraints
│   │   └── system_builder.py       # glue that takes schema instances and builds full B(T)y=0 set
│   │
│   ├── schemas/                    # law families & builder functions
│   │   ├── families.py             # SchemaFamily dataclass + SCHEMA_FAMILIES registry (S1–S11)
│   │   ├── s1_copy_tie.py          # build_S1_constraints(...)
│   │   ├── s2_component_recolor.py # build_S2_constraints(...)
│   │   ├── s3_bands.py             # build_S3_constraints(...)
│   │   └── ...                     # up to s11_...
│   │
│   ├── solver/
│   │   └── lp_solver.py            # wrapper around pulp / ortools: solve(constraints, N, C) -> Grid
│   │
│   ├── catalog/
│   │   ├── types.py                # SchemaInstance, TaskLawConfig dataclasses
│   │   └── store.py                # read/write catalog JSON/YAML (per-task law instances)
│   │
│   └── runners/
│       ├── run_single_task.py      # CLI: run full pipeline on one task_id using a given law config
│       └── sweep_training.py       # later: loop over training tasks, call Pi-agent, update catalog
│
└── tests/
    ├── test_features_coords.py
    ├── test_features_components.py
    ├── test_constraints_builder.py
    └── ... (small, focused tests)
```

---

* import from `src/core` / `src/features`,
* run small toy grids,
* verify outputs.

No god function, no huge files, everything composed by a thin runner later.

---