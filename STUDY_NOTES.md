# LLNL 2026 Data Science Challenge — Complete Study Notes
### Agentic AI for Materials Science

> Merged from my lecture notes + the repo README + inspection of the actual data files.
> Mentor: **Haichao Miao** (miao1@llnl.gov), Research Scientist at LLNL — Additive Manufacturing / 3D printing.
> Repo: `mhaichao/llnl_data_science_challenge_2026` (this repo is our fork/copy of it).

---

## 1. The Big Picture (one paragraph)

We 3D-print metal lattice parts using **laser powder bed fusion (LPBF)**. After printing, the only way to check the *inside* of the part without destroying it is an **X-ray CT scan**, which gives a 3D volume of density values. Today, a human materials scientist manually inspects that volume for defects — this is the **bottleneck** of the whole manufacturing pipeline, and LLNL wants it automated (eventually in real time). Our job: build an **agentic AI workflow** — an LLM equipped with tools (MCP), domain instructions (skills), and specialized workers (subagents) — that can segment the CT volume, extract the lattice structure, compare it against the intended design, find defects (missing / broken / thin / bent struts, dross), and write a professional **NDE (non-destructive evaluation) report**, all with traceable, reproducible outputs.

The tech stack progression from the kickoff talk: **OpenCV/image processing → LLM agents → materials science application (LPBF inspection)**.

---

## 2. Materials Science Background — Filling the Gaps

### What is LPBF?
**Laser Powder Bed Fusion**: a thin layer of metal powder is spread, a laser melts/fuses the cross-section of the part, the bed drops, a new powder layer is spread, repeat. It can build geometries impossible to machine — like internal lattices — but the process is failure-prone: incomplete fusion, warping, and missing/malformed features. Note the acronym is **LPBF** (my original notes had "LBPF" — the README itself typos it once too).

### What is a lattice structure?
Instead of printing solid metal, you fill the part's interior with a repeating **unit cell** — a small strut-based geometric motif tiled in 3D (here: 9×9×9 cells). You get most of the stiffness at a fraction of the weight.

### What is an octet unit cell, and *why are we so interested in it?* ✅ (open question answered)
An **octet cell** is a strut-based cell made of interconnected diagonal beams; tiled, it forms the **octet truss**. Reasons it's the star of this challenge:

1. **It's mechanically near-optimal.** The octet truss is *stretch-dominated*: under load, struts carry tension/compression along their axis (efficient) rather than bending (inefficient). This gives outstanding **stiffness-to-weight and strength-to-weight** ratios — my notes said "strength at low intensity"; the right word is **low density**.
2. **It's what LLNL actually prints.** Lightweight structures, energy absorption (crash/impact mitigation), thermal applications. The Part 2 dataset is a real LLNL/Tran et al. print in **Ti5553 titanium alloy**.
3. **It's a great inspection testbed.** The geometry is perfectly regular and known in advance, so a *missing* or *thin* strut is well defined — you can insert defects deliberately and measure whether your algorithm catches them.
4. **Defects matter here.** Because the truss is stretch-dominated, every strut is a load path. **A missing strut breaks a load path** — locally the structure loses strength disproportionately. That's why the defect classes are strut-level: missing, broken, thin, bent, plus **dross** (excess re-solidified material blobs, an LPBF-specific defect).

### What is X-ray CT data, concretely?
The object is rotated in an X-ray beam; material **absorbs** radiation and a reconstruction algorithm builds a **3D scalar field** — a 3D array where each **voxel** (3D pixel) holds a grayscale density value (in our simulated data, normalized to [0, 1]). High value ≈ metal, low value ≈ air. It is *not* a mesh or a point cloud — it's basically a 3D image, which is why image-processing tools (OpenCV, scikit-image) apply.

### Ground truth — correcting my earlier note
My notes said "GroundTruth is the CT scan." More precisely, it depends on the question:
- **For segmentation quality (Part 1, Task 7):** the ground truth is the provided reference mask image `data/9x9x9_octet_lattice/ground_truth_segmentation_slice_380.png`. We compare *our* segmentation of slice 380 against it.
- **For defect detection (Part 2):** the CT scan is the **measurement** (what was actually printed); the **design** (STL mesh and/or JSON graph of nodes+edges) is the **intent**. Defects = deviations of measurement from intent. So: align STL/JSON with the TIF, then find design struts with no material in the CT → missing struts.

---

## 3. The Analysis Pipeline (why each step exists)

The three-stage image pipeline, all shown on the same slice in `images/`:

| Stage | Input → Output | Why |
|---|---|---|
| **Segmentation** | density volume → binary mask (material=1, background=0) | Raw CT is fuzzy grayscale; thresholding turns "how dense is this voxel" into "is this metal or not." Simplest method: `mask = volume >= threshold`. Choosing the threshold well is the whole game (histograms, Otsu's method, visual feedback loops). |
| **Slicing** | 3D volume → one 2D cross-section | You can't eyeball a 3D array. A CT volume is literally a stack of 2D images; viewing slice *i* along an axis is how you (and the agent) inspect data and debug segmentations. |
| **Skeletonization** | binary mask → 1-voxel-thick centerline | See below. |

### *Why skeletonization?* ✅ (open question answered)
Skeletonization (`skimage.morphology.skeletonize`) erodes the segmented mask down to a **thin centerline that preserves shape and connectivity**. The point:

1. **It converts pixels into structure.** A blob of ~millions of foreground voxels becomes a sparse curve network whose branches ≈ **struts** and junction points ≈ **nodes**. That's the same vocabulary as the design's JSON graph (junctions + struts) — so now you can compare *printed topology* vs *designed topology* graph-to-graph.
2. **Defects become graph queries.** Missing strut = an edge in the design graph with no corresponding skeleton branch. Broken strut = a skeleton branch with a gap/disconnected component. Thin strut = small local radius of the mask around the skeleton centerline. Bent strut = centerline deviates from the straight line between its two junctions.
3. **It's something an LLM agent can reason about.** An agent can't "look at" 10⁸ voxels, but it *can* reason over "junction 412 has 11 of 12 expected struts." Segmentation + skeletonization turn density into a symbolic structure that fits in a context window. (This was the punchline of the lecture: *"segmentation and skeletonization turn density into a structure an agent can reason about."*)

---

## 4. *What's the role of AI in materials science?* ✅ (open question answered)

- **Old paradigm (traditional ML):** isolated point tasks — predict one property, segment one image. A human still glues every step together.
- **New paradigm (agentic AI):** materials science is now **data-rich and workflow-heavy** — the bottleneck isn't a single prediction, it's running the whole multi-step analysis. Agents = LLMs + tools + planning + environment access = **autonomous AI research assistants** that execute the entire workflow.
- **The agent loop** (memorize this — it's the mental model for everything we build):

  > **Plan → write & call code → inspect artifacts (slices, masks, reports, logs, failures) → revise → repeat**

  The "inspect artifacts" step is what makes it *agentic* rather than a script: the agent looks at its own outputs (e.g., a rendered slice of its segmentation) and self-corrects.

---

## 5. Agentic AI Building Blocks (the four pillars)

### 5.1 MCP — Model Context Protocol
The standardized way to let an LLM call our Python functions. Before MCP, every tool × every LLM needed custom glue code; MCP is one open protocol.

- **Architecture:** *MCP server* (our Python script hosting the tools) ↔ *MCP client* (Codex CLI) speaking **JSON-RPC**.
- **Why it's good for science:** (1) decouples our scientific code from the LLM, (2) tools are auto-discovered, (3) security — a controlled layer defining exactly what the LLM may do.
- **We use FastMCP.** The rules that matter (this is what the "well-written descriptions" note was about — the docstring *is* the tool's UI for the agent):
  - `@mcp.tool()` exposes a function; the **function name becomes the tool name**.
  - The **docstring becomes the tool description** — write it for the agent: what it does, what each arg means, what it returns.
  - **Type annotations define the input schema** (`threshold: float`); params **without defaults are required**, with defaults are optional (`axis: int = 0`).
  - **No `*args`/`**kwargs`** — FastMCP needs an explicit signature to build the JSON schema.
  - Return short, useful status strings (or structured data when needed downstream).
- **Registration:** add the server to `~/.codex/config.toml` under `[mcp_servers.<name>]` with absolute paths to the Python exe and `src/mcp_server.py`. **Restart Codex CLI after every config change** (it does not hot-reload). Verify with `/mcp`.

### 5.2 Skills
MCP's weakness: many tools up front **bloats the context window** and hurts **tool selection** reliability. A **skill** is a focused, on-demand instruction package (a `SKILL.md` + optional scripts) for a specific workflow — loaded only when relevant. My one-liner from the talk holds: *"skills package domain instructions so the agent writes like a materials scientist."*

The provided `nde_report_expert` skill (`.agents/skills/nde_report_expert/`) demonstrates all three capabilities a skill can have:
1. Runs a local script (`scripts/3d_visualize.py`, called twice at elev/azim 30°/45° and 60°/45°),
2. Autonomously invokes our MCP tools (`segment_ct_dataset`, `skeletonize`) if intermediate files don't exist,
3. Carries system instructions for report structure (summary metrics table, visual gallery, analysis section) and hygiene rules (check array shapes, clean up temp scripts).

Skills only load if Codex CLI is **started from the repo root** (it looks for `.agents/skills/`), and — same as MCP — **restart after adding/editing a skill**.

### 5.3 Subagents

