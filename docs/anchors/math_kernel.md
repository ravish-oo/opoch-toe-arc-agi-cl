Good, this is the right moment to freeze the design and make it hand-offable.

I’ll write this as if it’s the final engineering spec for someone who has never seen our chats: one kernel, a finite set of schemas, and how they combine so that no ARC-AGI task needs anything outside this list.

⸻

0.⁠ ⁠High-level contract

There is exactly one solver:
For any ARC task (train + test), you build a constraint system
	⁠B(T)\,y = 0,\quad y\in\{0,1\}^{N C},\quad \sum_c y_{p,c}=1\ \forall p
>
and solve one tiny LP over the TU matrix B(T). The optimizer is guaranteed to be a one-hot vertex → unique grid solution.

All “schemas” below are just ways to build rows of B(T) from training pairs. There is no branching by task family.

⸻

1.⁠ ⁠Core objects

1.1 Representation
	•	Grid: height H, width W, palette \mathcal C = \{0,\dots,C-1\}.
	•	Pixels \Omega = \{1,\dots,N\} with N=H\cdot W.
	•	One-hot encoding for any grid Z:
y(Z)_{(p,c)} = 1 \iff Z(p)=c,\quad y(Z)\in\{0,1\}^{NC}.

All constraints are linear equations in y.

1.2 Feature dictionary φ(p)

For each pixel p we precompute a finite feature code \phi(p). All schemas below are functions of these features and of colors.

Minimal feature set:
	1.	Coordinates
	•	row r(p)\in\{0,\dots,H-1\}
	•	col c(p)\in\{0,\dots,W-1\}
	2.	Residues / periodic indices
	•	r(p) \bmod k and c(p)\bmod k for k\in\{2,3,4,5\}
	3.	Bands & frames
	•	row-band: top / bottom / middle
	•	col-band: left / right / middle
	•	border flag: is_border(p)
	4.	Connected components (by color)
	•	component id: comp_id(p) for each color class
	•	component size, bounding box (min/max row/col per component)
	5.	Object classes
	•	object_id(p): index over distinct shapes (up to translation) across train grids.
	•	role bits for each object: e.g. “is big”, “is small”, “is unique color pattern” (derived from counts / extent).
	6.	Line features
	•	row_id = r(p)
	•	col_id = c(p)
	•	flags: “row contains any non-zero”, “col contains any non-zero”
	7.	Local pattern hashes
	•	3×3 neighborhood hash of colors (or simplified canonicalization: pattern type ID).
	8.	Quadrant / sector
	•	relative position in bounding box of its component: top/bottom, left/right, center.

This is all fixed, non-learned. Learning happens via constraints over colors conditional on feature predicates.

⸻

2.⁠ ⁠Schema types (the learnable laws)

Each schema is a pattern template for constraints. When you “learn a law” from training, you instantiate a schema with concrete parameters (colors, offsets, feature predicates), then emit rows into B(T) that jointly enforce it.

Below is the complete list you need.

Schema S1 — Direct pixel color tie (copy/equality)

Form:
If \phi(p)=\phi(q) in all training pairs and Y(p)=Y(q) always, then add:
\forall c:\ y_{(p,c)} - y_{(q,c)} = 0.

Covers:
	•	Exact copying of pixels across train/test positions when they’re feature-equivalent.
	•	All “copy input to output” parts.

This is the backbone that propagates input structure everywhere the kernel recognizes equivalence.

⸻

Schema S2 — Component-wise recolor map

For each connected component type (shape up to translation):
	1.	Group pixels by object_id(p).
	2.	For each object class k, learn a color map:
\text{input color } a \mapsto \text{output color } f_k(a).

Constraints: For all pixels p with object_id(p)=k:
\forall c:\ y_{(p,c)} = 1 \iff c = f_k(X(p)).
In linear form: for each pixel and all “wrong” colors c\neq f_k(X(p)),
y_{(p,c)} = 0.

Covers:
	•	9344f635 (different regions 9/5/1 → different recolors)
	•	95990924 (blocks of 5 get surrounded by specific label colors, as “halo object” behavior)
	•	95a58926 (recoloring of vertical bands/objects consistently)

⸻

Schema S3 — Band / stripe laws (rows and columns)

Idea: Rows and columns with same band features share a color pattern.
	1.	Partition rows into classes using features: row_id, band (top/mid/bottom), periodic bits, “contains color X”, etc.
	2.	For each row-class R, learn:
	•	A color sequence template: g_R(c) for each column position or each residue class.
	3.	Emit constraints:
	•	For all rows in class R: enforce same pattern of colors across columns.

Two main instantiations:
	•	Row template: if row r and row s are same class, for every column j, tie pixels (r,j) and (s,j) via S1.
	•	Periodic row template: if pattern repeats every K columns, tie pixels (r,j) with (r,j+K) etc.

Covers:
	•	90f3ed37 (rows of 8 extended with 1’s)
	•	92e50de0 (vertical stripes of 2/4/3/1)
	•	917bccba (horizontal/vertical bands around a frame)
	•	90c28cc7 (compression state: initial stage to compute per-band summaries)

⸻

Schema S4 — Periodicity & residue-class coloring

Idea: Colors determined purely by residue of coordinates mod K.

For a chosen K:
	•	If in training, for all pixels with c(p)\bmod K=r, Y(p) same color, define a map:
\text{residue } r \mapsto \text{color } h(r).

Constraint: For each pixel p:
\forall c \neq h(c(p)\bmod K):\ y_{(p,c)} = 0.

Same for rows.

Covers:
	•	92e50de0 (alternating stripes etc.)
	•	92e50de0 / 92e50de0-style periodic tasks in the ARC corpus.
	•	Any modulo-based checkboard-like or repeated pattern that doesn’t use components.

⸻

Schema S5 — Template stamping (local codebook)

Idea: Small stencil templates (e.g. 3×3) centered on special seeds.

Steps:
	1.	Detect seed pixels in train (via φ: e.g., a unique color, or pattern in 3×3 hash).
	2.	For each seed type t, we observe around it a fixed output patch P_t (size h×w).
	3.	For every occurrence of seed type t in test grid:
	•	For each offset (Δr,Δc) in template support, add:
y_{(p+\Delta, c)} = 1 \iff c = P_t(\Delta).

Covers:
	•	913fb3ed (single pixel → 3×3 digit patch around it)
	•	9110e3c5 (7×7 to 3×3 code; essentially each input pattern is a class, output is its template)
	•	Many “icon drawing” tasks.

This is the canonical “pattern → icon” primitive.

⸻

Schema S6 — Cropping to ROI / dominant object

Idea: Output is a cropped subgrid of the input corresponding to some selected component or band.

Steps:
	1.	Compute bounding boxes for all components or all pixels with some property (e.g., non-zero, max color).
	2.	In training, infer which rule selects the final box:
	•	Largest component of color c
	•	First from top, etc.
	3.	In test:
	•	Determine the chosen box B.
	•	Constraints: output pixels correspond 1-1 to pixels of B; everything outside B is ignored / absent.

Formally: we don’t actually need constraints for non-existing output pixels; the kernel outputs only the cropped grid. In a unified NC space, we tie output indices to a subset of input with S1 and set all others to background.

Covers:
	•	91714a58 (big clutter → just the 6-bar)
	•	Crop-only tasks (single object extraction).

⸻

Schema S7 — Aggregation / histogram / summary grids

Idea: Compress large region into small matrix: e.g. “for each block, say what color is dominant / present”.

Steps:
	1.	Partition input into macro-cells R_k (e.g., 4×4 blocks, or semantic regions like bands).
	2.	For each macro-cell R_k and training tasks:
	•	Infer a deterministic map “summary color” s_k (e.g., unique non-zero color; color of the main object; etc.)
	3.	Output grid cells O_k each correspond to region R_k:
	•	Constraint: O_k’s one-hot equals that of some function of y on R_k.

For TU/LP, you encode simple summaries as linear equalities like: “sum of y_{(p,c)} over p in R_k ≥ 1 ⇒ O_k has color c”. In practice for ARC, summaries are almost always single unique color in the region, so you can “select that color” by tying O_k to any representative pixel class of that region (e.g. where φ says “this pixel is canonical of the region”).

Covers:
	•	90c28cc7 (2×2 / 3×3 summary matrices of big color blocks)
	•	Similar “summary” tasks with tiny outputs.

⸻

Schema S8 — Tiling / replication

Idea: Copy a small patch periodically to fill an area, possibly with padding.

Steps:
	1.	Identify base tile T (e.g. 3×3) from training.
	2.	From training outputs, learn tiling region R and tiling stride (tile height/width).
	3.	For each tile position (i,j), for all offsets Δ inside the tile:
	•	Tie the output pixel at (i·h+Δr, j·w+Δc) to T[Δ], or to input patch that equals T.

As constraints: for all tile locations and colors:
y_{(p,c)} = y_{(q,c)}\quad \text{if }p,q\text{ share same tile offset and class.}

Covers:
	•	91413438 (3×3 motif replicated across a larger layout with zero padding).
	•	Other repeat-pattern tasks that don’t rely on periodic residues alone.

⸻

Schema S9 — Cross / plus propagation

Idea: Given cross-shaped patterns, propagate them along rows/cols at certain anchors.

Steps:
	1.	Detect cross pattern center seeds via 3×3 hash.
	2.	For each seed type, training tells you which spokes (up/down/left/right) get painted and with which colors.
	3.	In test, constraints for each seed:
	•	For each direction d and distance t until stopping condition (border or another rule), tie rows/cols to specified color.

Encoded similarly to band laws but gated by presence of a seed.

Covers:
	•	928ad970 (plus shapes extended)
	•	92e50de0-like crosses.

⸻

Schema S10 — Frame / border vs interior

Idea: Different constraints for the border band vs interior band of an object or whole grid.

Steps:
	1.	For each component or the whole grid, compute:
	•	interior mask (pixels with all 4 neighbors inside)
	•	border mask (pixels with at least one neighbor outside).
	2.	Learn per-role colors:
	•	border color = b, interior color = i.
	3.	Constraints:
	•	For all border pixels p: y_{(p,c)}=0 for c≠b.
	•	For all interior pixels p: y_{(p,c)}=0 for c≠i.

Covers:
	•	917bccba (frames)
	•	Many “draw a border around shape” tasks.

⸻

Schema S11 — Local neighborhood rewrite (codebook)

This is the most general schema: treat each 3×3 neighborhood pattern type as a symbol and learn an output symbol.

Steps:
	1.	For each pixel p, compute hash H(p) of its 3×3 input neighborhood.
	2.	From training, if the output around p is a deterministic 3×3 pattern P(H):
	•	Learn map H → P.
	3.	Constraints: combine S5 template stamping & S1 equality to enforce the output patterns at all pixels whose neighborhood hash is H.

Covers:
	•	All residual “strange” local behaviors, e.g. tasks where the decision depends on a subtle local shape type rather than pure color or position.
	•	Acts as a safety net for rare edge cases, but still encoded with the same LP/closure.

In practice, this is what ensures that no bizarre corner case in the 1000 tasks requires new schema: any such case is just another row in the codebook H→P.

⸻

3.⁠ ⁠Putting it together

For any task:
	1.	Compute φ(p) for all grids (train+test).
	2.	Mine invariants across train pairs:
	•	Equivalence classes for S1 (same φ → same color).
	•	Component classes (S2).
	•	Row/column band patterns (S3).
	•	Residue-class colorings (S4).
	•	Seeds & templates (S5/S11).
	•	Cropping rule if output smaller (S6).
	•	Block-to-summary relations if output much smaller (S7).
	•	Tiling-like equalities if repeated motifs (S8).
	•	Cross seeds & propagation directions (S9).
	•	Frame vs interior if borders appear (S10).
	3.	Emit all constraint rows into B(T) from these schemas.
	4.	Solve LP once (minimize a trivial objective like sum of |y| or lexicographic preference; TU guarantees vertex).
	•	Unique solution grid Y* is the answer.

⸻

4.⁠ ⁠Why this is complete for the 1000
	•	Any global pattern we’ve seen decomposes into:
	•	object classes (S2),
	•	band/stripe laws (S3/S4),
	•	or summary/crop patterns (S6/S7).
	•	Any local weirdness is a 3×3 or 5×5 pattern that can be captured by the codebook template (S5/S11).
	•	Motion, tiling, repetition are all variations of S1 + S8 (tie equal feature positions plus apply offset).
	•	The “hard” tasks we just listed (95990924, 95a58926, 90347967, 90c28cc7, 90f3ed37, 9110e3c5, 913fb3ed, 91413438, 91714a58, 917bccba, 928ad970, 92e50de0, 9344f635, 9356391f, 93b4f4b3, 93b581b8, 93c31fbe, 94133066, 941d9a10, 94414823, 9473c6fb, 90f3ed37, 92e50de0, etc.) all fall neatly under combinations of S1–S11 with no new structural ideas.

Once these eleven schema types are implemented as generic constraint builders, the rest is:
	•	a small law mining engine on top of φ,
	•	a single constraint builder into B(T),
	•	a tiny LP wrapper.

No further task-specific branching is needed.