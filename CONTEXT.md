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
The sampling-time operation that blocks a reverse edge deletion when it would violate an active constraint target.
_Avoid_: Training constraint, conditioning

**Joint projection**:
Projection enforcement against the conjunction of all active constraint targets.
_Avoid_: Sequential projection, combined model

**Projection profile**:
One point in the cycle-count by cycle-length generation grid, describing which constraint targets were enforced while producing a sample set.
_Avoid_: Metric configuration, model variant

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

**Absorbing edge label**:
An edge category toward which every pair label converges at the terminal point of an absorbing edge diffusion.
_Avoid_: Terminal bond, noise bond

**Positive-marginal edge insertion**:
A topology-insertion diffusion whose absorbing distribution spans present-edge labels in their empirical proportions.
_Avoid_: Standard insertion, random-edge insertion

**Single-bond edge insertion**:
A topology-insertion diffusion whose sole absorbing edge label is the single-bond category.
_Avoid_: Pure insertion, symmetric insertion
