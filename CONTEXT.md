# Constrained Molecular Graph Generation

This context describes how a single graph diffusion model is evaluated under configurable structural requirements.

## Language

**Generalist checkpoint**:
A checkpoint trained on the complete dataset without a structural projection or structural filtering.
_Avoid_: Unconstrained checkpoint, universal model

**Constraint target**:
A structural predicate and threshold measured for every generated graph, whether or not it is enforced.
_Avoid_: Projector setting, condition

**Projection enforcement**:
The sampling-time operation that blocks a reverse topology mutation—edge insertion or edge deletion—when it would violate an active constraint target.
_Avoid_: Training constraint, conditioning

**Forward diffusion modality**:
The topology corruption family used in training: edge addition moves pairs toward present-edge labels, while edge deletion moves pairs toward the no-edge label.
_Avoid_: Generation direction, projector direction

**Reverse denoising operation**:
The topology mutation performed during generation: deletion for the edge-addition modality and insertion for the edge-deletion/edge-absorbing modality.
_Avoid_: Forward transition, modality

**Telemetry cohort**:
A deterministic seed-derived subset of global generated-graph indices whose complete denoising states are retained for trajectory and chemical repair analysis.
_Avoid_: Sampling batch, validation subset

**Constrained dead-end state**:
A denoising state from which no molecule valid under the established RDKit largest-fragment policy is reachable by any finite sequence of reverse-kernel-supported mutations while every active projection constraint remains satisfied. Profiles without effective projection constraints have no dead-end classification.
_Avoid_: One-edit failure, constraint violation, permanently invalid molecule

**Dead-end onset**:
The earliest monitored state on a final-invalid denoising trajectory that is a constrained dead end. Later states remain dead ends because the reverse process can only reduce the set of states reachable from the trajectory.
_Avoid_: First invalid state, first blocked action

**Joint projection**:
Projection enforcement against the conjunction of all active constraint targets.
_Avoid_: Sequential projection, combined model

**Projection profile**:
One point in the cycle-count by cycle-length generation grid, describing which constraint targets were enforced while producing a sample set.
_Avoid_: Metric configuration, model variant

**Example molecule**:
A reproducibly selected molecule that is valid under the established RDKit largest-fragment policy and is exported from one projection profile for qualitative inspection. It is illustrative rather than statistically representative of that profile.
_Avoid_: Representative molecule, generated batch

**Cross-evaluation target**:
A structural predicate applied post hoc to every sample set, independently of its projection profile.
_Avoid_: Projector, generation constraint

**Matrix evaluation**:
The comparison of every projection profile against the same complete set of cross-evaluation targets.
_Avoid_: Grid search, hyperparameter sweep

**Validation reference profile**:
A conjunction of structural predicates that selects a QM9 validation cohort for distributional comparison.
_Avoid_: Filtered dataset, projection profile

**FCD cross-matrix**:
The comparison of every generated projection profile against every validation reference profile using Fréchet ChemNet Distance.
_Avoid_: Matched FCD, conditional generation score

**Graph cycle summary**:
The ordered pair comprising the number of all unique undirected simple cycles and the maximum such cycle length for one graph. Acyclic graphs have the summary `(0, 0)`.
_Avoid_: Ring count, cycle rank, minimum cycle basis

**Processed dataset order**:
The stable graph-index order of a persisted dataset split, independent of any epoch-specific batching or loader shuffling.
_Avoid_: Training order, batch order

**Absorbing edge label**:
An edge category toward which every pair label converges at the terminal point of an absorbing edge diffusion.
_Avoid_: Terminal bond, noise bond

**Positive-marginal edge insertion**:
A topology-insertion diffusion whose absorbing distribution spans present-edge labels in their empirical proportions.
_Avoid_: Standard insertion, random-edge insertion

**Single-bond edge insertion**:
A topology-insertion diffusion whose sole absorbing edge label is the single-bond category.
_Avoid_: Pure insertion, symmetric insertion

**Blocked action**:
Every projector-rejected reverse topology proposal, including repeated proposals of a graph-edge pair already cached as blocked.
_Avoid_: Failed denoising update

**Newly blocked graph-edge pair**:
The first rejected proposal for one undirected edge in one graph; subsequent rejected proposals for that pair are repeated blocked actions.
_Avoid_: Blocked action count

**Conditional positive-class sampling**:
Choosing a present-edge class from an edge distribution conditioned on edge presence when a deletion is blocked.
_Avoid_: Forced single bond, chemical repair
