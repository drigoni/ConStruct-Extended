# Methodology and paper notes: cycle-constrained molecular graph diffusion

> Repository snapshot: commit `e94cc36203d2af3faeac03cacdc47e5653dc8a70`
> (2026-09-22). These are technical notes for preparing a paper, not a paper
> draft. Numerical claims below are tied to the saved artifacts named in the
> provenance section.

## 1. Executive summary

This repository extends ConStruct with cycle-count and maximum-cycle-length
constraints for molecular graph generation. A generalist discrete graph
diffusion model is trained on full, unfiltered QM9, and structural constraints
are imposed only at sampling time by a greedy reverse-process projector. Two
monotone edge processes are studied:

1. **Forward edge deletion / reverse edge addition** (`absorbing_edges`). The
   terminal graph is empty. During denoising, a projector rejects proposed edge
   additions that would violate an upper bound such as “at most $K$ cycles”
   or “maximum cycle length at most $L$.” This is the direction originally
   supported by ConStruct for edge-deletion-invariant properties.
2. **Forward edge insertion / reverse edge deletion** (`edge_insertion`). The
   terminal graph is complete. During denoising, a projector rejects proposed
   edge deletions that would violate a lower bound such as “at least $K$
   cycles” or “maximum cycle length at least $L$.” This is the main extension
   explored here.

The saved 10,000-sample, seed-0 QM9 matrices give a crisp empirical message:

- The **edge-deletion diffusion is robust in the evaluated matrix**. Every
  enforced upper-bound target has 100% structural satisfaction, while molecular
  validity is 99.05–99.72% (98.76% without projection).
- The **edge-insertion diffusion enforces the requested topology but can damage
  chemistry**. Every enforced lower-bound target also has 100% structural
  satisfaction, but validity drops from 98.19% without projection to 90.62%
  for at least one cycle, 85.51% for a cycle of length at least four, 80.02%
  for length at least five, and 61.62% for the joint “at least two cycles and
  maximum length at least five” profile.
- Therefore the correct claim is not that insertion projection fails. It
  **succeeds structurally and fails increasingly often chemically**. The
  projector checks binary topology only; it neither enforces valence nor
  optimizes bond order.
- A boundary audit supplies a plausible mechanism. Among final graphs with
  exactly one cycle, deletion-diffusion samples constrained to at most one
  cycle are 99.69% valid (9,259 graphs), whereas insertion-diffusion samples
  constrained to at least one cycle are 84.53% valid (5,456 graphs), despite
  similar mean edge counts (8.81 versus 8.73). A blocked deletion in the latter
  process retains an edge the model proposed to remove, which can preserve a
  stale or chemically incompatible bond.
- Conditional resampling of the positive bond class for a blocked deletion is
  implemented, but its 10,000-sample matrix does **not** show a uniform
  improvement: validity changes range from $-0.46$ to $+1.05$ percentage
  points across constrained profiles. It is a useful ablation, not a solved
  insertion method.

These results are substantial but preliminary: the matrices use one training
seed and one sampling seed, the deletion and original insertion sample sets
predate a later clean-endpoint correction, and the two directions use different
checkpoints. A paper should present the saved numbers as the current evidence,
not as a multi-seed general theorem.

## 2. Relationship to ConStruct and the contribution in this repository

The original ConStruct method (Madeira, Vignac, Thanou, and Frossard, NeurIPS
2024) introduces an edge-absorbing discrete diffusion and a projector for
properties invariant under edge deletion. Its forward process deletes edges;
its reverse process adds candidate edges and rejects those that violate the
property. See the [published paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f82385b8804009f9a81e1a30f1ff14e3-Abstract-Conference.html)
and [official implementation](https://github.com/manuelmlmadeira/ConStruct).

This fork contributes or investigates:

- all-simple-cycle count and maximum-simple-cycle-length constraints;
- upper-bound projectors paired with forward edge deletion;
- the opposite, forward edge-insertion process paired with lower-bound
  projectors;
- joint count-and-length projection;
- one generalist checkpoint per transition, with targets selected at sampling
  time rather than one target-filtered model per condition;
- feasibility-conditioned node-count sampling for lower-bound constraints;
- post-hoc structural cross-evaluation of immutable sample sets;
- a 9-by-9 FCD cross-matrix against structurally filtered validation cohorts;
- clean-endpoint diagnostics and a checkpoint-compatible sampler correction;
- optional conditional positive-class resampling when an edge deletion is
  blocked.

An important theoretical qualification is that the original ConStruct
distribution-matching argument assumes clean training graphs satisfy the target
property. This fork deliberately trains on **full, unfiltered QM9** and changes
the target at inference time. The projector still provides a pathwise hard
topological guarantee, but the projected distribution should not be described
as the learned conditional distribution $p_{\mathrm{data}}(G\mid G\in \mathcal C)$, nor does the original target-specific distribution-matching
argument automatically apply.

## 3. Graph representation and notation

Let a molecular graph be

\[
G=(X,C,E,N),
\]

where $N$ is the number of active heavy-atom nodes, $X\in \{0,1\}^{N\times d_X}$ contains categorical atom types, $C\in \{0,1\}^{N\times d_C}$ contains categorical formal charges, and

\[
E\in\{0,1\}^{N\times N\times d_E},\qquad E_{ij}=E_{ji},\quad E_{ii}=0,
\]

contains categorical edge labels. For QM9 without explicit hydrogens, atom
types are the usual heavy atoms $\{\mathrm C,\mathrm N,\mathrm O,\mathrm F\}$.
Edge class 0 is `NO_EDGE`; positive classes encode single, double, triple, and
aromatic bonds. A Boolean adjacency matrix is obtained by discarding bond
order:

\[
A_{ij}(G)=\mathbb 1[E_{ij}\ne 0].
\]

The projector operates on $A(G)$, not on the full categorical edge tensor.
This distinction is central to the insertion failure mode.

Variable graph sizes are represented with a node mask. At generation time,
$N\sim p_N$, the empirical QM9 node-count distribution. Under lower-bound
constraints this distribution is truncated and renormalized to feasible sizes:

\[
p_N^{(n_{\min})}(n)
=\frac{p_N(n)\mathbb 1[n\ge n_{\min}]}
       {\sum_m p_N(m)\mathbb 1[m\ge n_{\min}]}.
\]

For a length lower bound $L$, $n_{\min}\ge L$. For a cycle-count lower
bound $K$, the implementation chooses the smallest supported $n$ for which
the complete graph $K_n$ has at least $K$ unique simple cycles.

## 4. Discrete denoising diffusion

### 4.1 General categorical transition

Each categorical feature family $M\in\{X,C,E,Y\}$ is diffused independently
conditional on the graph mask. Let $d_M$ be its number of classes and let
$\boldsymbol\pi_M\in\Delta^{d_M-1}$ be its terminal distribution. Define the
rank-one matrix

\[
P_M=\mathbf 1\boldsymbol\pi_M^\top,
\]

whose rows are all $\boldsymbol\pi_M$. The one-step transition is

\[
Q_t^M=(1-\beta_t^M)I+\beta_t^M P_M,
\]

and, because $P_M^2=P_M$, the cumulative transition is

\[
\bar Q_t^M=\bar\alpha_t^M I+(1-\bar\alpha_t^M)P_M,
\qquad
\bar\alpha_t^M=\prod_{r\le t}(1-\beta_r^M).
\]

For a one-hot clean label $m_0$, direct noising is therefore

\[
q(m_t\mid m_0)=\operatorname{Cat}(m_0\bar Q_t^M).
\]

Training samples $t\sim\mathcal U\{1,\ldots,T\}$, with $T=500$ in the
large QM9 experiments, and draws $G_t\sim q(G_t\mid G_0)$ directly.

Nodes and charges use empirical marginal terminal distributions and the cosine
schedule in both specialized edge models. Edges use the linear absorbing
schedule

\[
\beta_r=\frac{1}{T-r+1},\qquad r=0,\ldots,T,
\]

under the implementation's array indexing. Hence, for $r<T$,

\[
\bar\alpha_r=\prod_{k=0}^{r}(1-\beta_k)
=\frac{T-r}{T+1},
\]

and $\bar\alpha_T=0$. The endpoint is exactly the selected edge terminal
distribution.

### 4.2 Forward edge deletion (`absorbing_edges`)

Let class 0 denote `NO_EDGE`. The edge terminal distribution is

\[
\boldsymbol\pi_E^{\mathrm{del}}=\delta_0.
\]

Thus

\[
Q_{t,E}^{\mathrm{del}}
=(1-\beta_t)I+\beta_t\mathbf 1\delta_0^\top.
\]

A no-edge remains a no-edge, while every present bond either retains its label
or is replaced by no-edge. Forward trajectories are monotonically decreasing
in adjacency:

\[
A(G_t)\subseteq A(G_{t-1}),
\]

and $G_T$ is an empty graph over the sampled nodes. The reverse sampler is
therefore an edge-addition process.

### 4.3 Forward edge insertion (`edge_insertion`)

Let $m_E(k)$ be the empirical edge-label marginal. The positive-marginal
terminal distribution is

\[
\pi_E^{\mathrm{ins}}(0)=0,\qquad
\pi_E^{\mathrm{ins}}(k)
=\frac{m_E(k)}{\sum_{j=1}^{d_E-1}m_E(j)},\quad k>0.
\]

The transition is

\[
Q_{t,E}^{\mathrm{ins}}
=(1-\beta_t)I+\beta_t\mathbf 1
(\boldsymbol\pi_E^{\mathrm{ins}})^\top.
\]

A present bond remains present, although its positive class can change; a
no-edge can become a positive bond. Hence

\[
A(G_{t-1})\subseteq A(G_t),
\]

and $G_T$ is a complete graph whose off-diagonal bond classes are sampled
from the positive marginal. The reverse sampler deletes edges. The
`edge_insertion_single` control replaces the positive marginal by
$\delta_{\mathrm{SINGLE}}$; it changes the terminal bond-label distribution,
not the complete terminal topology.

The deletion and insertion channels are not symmetric on sparse molecular
data. From the saved QM9 marginal (72.61% no-edge), an exact categorical
calculation at $\bar\alpha=0.5$ gives edge-label mutual information of 0.3925
bits for deletion, 0.2648 bits for positive-marginal insertion, and 0.3145 bits
for single-bond insertion. This is evidence of channel asymmetry, not by itself
evidence that mutual information causes the final validity gap.

### 4.4 Denoiser and training loss

The network $f_\theta(G_t,t)$ is a graph Transformer that produces logits for
the clean node, charge, and edge labels. Each layer jointly updates node, edge,
and graph-level representations. Edge features modulate node attention, and
global features modulate both node and edge updates. Edge outputs are explicitly
symmetrized.

The large runs use 9 layers; hidden node, edge, and global dimensions are all
128; there are 4 attention heads; and feed-forward dimensions are 256. Inputs
include the noisy categorical state and time, spectral features, local cycle
features, degree/adjacency-derived features, and molecular charge, valency, and
weight features. See [the training configs](configs/experiment/training/) and
[model implementation](ConStruct/models/transformer_model.py).

The network predicts the clean state rather than the previous noisy state:

\[
\hat p_\theta(M_0\mid G_t,t)
=\operatorname{softmax}(f_\theta^M(G_t,t)).
\]

The training objective is weighted categorical cross-entropy over active nodes
and off-diagonal active node pairs:

\[
\mathcal L_{\mathrm{train}}
=\lambda_X\operatorname{CE}(X_0,\hat p_\theta^X)
+\lambda_C\operatorname{CE}(C_0,\hat p_\theta^C)
+\lambda_E\operatorname{CE}(E_0,\hat p_\theta^E)
+\lambda_Y\operatorname{CE}(Y_0,\hat p_\theta^Y),
\]

with $(\lambda_X,\lambda_C,\lambda_E,\lambda_Y)=(1,2,5,0)$. Validation also
computes a variational bound consisting of the node-count log probability, the
terminal-prior KL, and diffusion posterior KL terms. The large checkpoints were
selected using sampled molecular validity with validation checks every 10
epochs and a minimum of 400 epochs.

### 4.5 Reverse posterior

For one categorical variable, let $i,j,k$ denote values of $M_0,M_t,M_s$,
respectively, with $s<t$. The exact forward posterior is

\[
q(M_s=k\mid M_t=j,M_0=i)
=\frac{[\bar Q_s]_{ik}[Q_{s\to t}]_{kj}}
       {[\bar Q_t]_{ij}}.
\]

The learned reverse kernel marginalizes this posterior using the predicted
clean-label distribution:

\[
p_\theta(M_s=k\mid G_t)
=\sum_i q(M_s=k\mid M_t=j,M_0=i)
\hat p_\theta(M_0=i\mid G_t,t).
\]

Nodes, charges, and the upper triangle of the edge tensor are sampled from these
categorical distributions; the sampled edge tensor is mirrored to preserve
undirected symmetry.

### 4.6 Clean-endpoint correction

The implementation's linear schedule has
$\bar\alpha_0=T/(T+1)$, rather than 1. Historically the reverse loop returned
array index zero as if it were clean, leaving residual corruption. For
$T=500$, a clean absent edge under the insertion endpoint acquired a spurious
edge with probability $1/501$. With $m$ absent pairs, the probability of at
least one such corruption is

\[
1-\left(\frac{500}{501}\right)^m,
\]

which is about 5.44% at $m=28$. This is a corruption probability, not an
invalidity probability.

The corrected final reverse step targets clean data by setting

\[
\bar Q_s=I,\qquad Q_{s\to t}=\bar Q_t
\]

when $s=0$, while retaining the trained schedule and the same denominator.
No retraining is required. In a paired 1,024-sample diagnostic, this changed
insertion validity from 93.26% to 96.97% (38 rescued, 0 harmed) and single-bond
insertion from 95.61% to 97.66% (21 rescued, 0 harmed); deletion stayed 98.44%
in aggregate (1 rescued, 1 harmed). The diagnostic isolates a real sampler bug,
but it does not explain every remaining directional difference.

## 5. Cycle properties

Let $\mathcal S(G)$ be the set of all unique undirected simple cycles of
$G$. A cycle is canonicalized over rotations and reversal so that each
undirected simple cycle is counted once. The repository defines

\[
c(G)=|\mathcal S(G)|
\]

and

\[
\ell(G)=
\begin{cases}
0,&\mathcal S(G)=\varnothing,\\
\max_{\gamma\in\mathcal S(G)}|\gamma|,&\text{otherwise}.
\end{cases}
\]

This must be called **all-simple-cycle count**, not cycle rank, circuit rank,
minimum cycle basis size, SSSR count, or RDKit ring count. The quantities can
differ dramatically. One audited 9-node QM9 topology has 14 unique simple
cycles but cycle rank 4. Cycle enumeration can be exponential in dense graphs;
threshold predicates stop early where possible, but the definition remains all
simple cycles without a length cap.

The feasible sets are

\[
\mathcal C_{\le K}^{c}=\{G:c(G)\le K\},\qquad
\mathcal C_{\le L}^{\ell}=\{G:\ell(G)\le L\},
\]

\[
\mathcal C_{\ge K}^{c}=\{G:c(G)\ge K\},\qquad
\mathcal C_{\ge L}^{\ell}=\{G:\ell(G)\ge L\}.
\]

Joint profiles use the conjunction, not sequential independent projection:

\[
\mathcal C_{\le K,\le L}
=\mathcal C_{\le K}^{c}\cap\mathcal C_{\le L}^{\ell},
\qquad
\mathcal C_{\ge K,\ge L}
=\mathcal C_{\ge K}^{c}\cap\mathcal C_{\ge L}^{\ell}.
\]

Upper-bound predicates are downward closed under edge deletion. Lower-bound
predicates are upward closed under edge addition. These monotonicities make
them compatible with the two opposite absorbing processes.

## 6. Reverse-process projection

The operator called a projector is a greedy feasibility filter, not an
orthogonal projection in a vector space. Let $G_t$ be the previously accepted
state and $\widetilde G_s\sim p_\theta(G_s\mid G_t)$ the model proposal.

### 6.1 Upper-bound projector after forward edge deletion

The terminal graph is empty and reverse proposals add edges. Let

\[
D^+=E(\widetilde G_s)\setminus E(G_t)
\]

be the candidate additions. Starting from $H\leftarrow G_t$, process each
$e\in D^+$ and set

\[
H\leftarrow
\begin{cases}
H+e,&\phi(H+e)=1,\\
H,&\phi(H+e)=0,
\end{cases}
\]

where $\phi$ is the active count, length, or joint predicate. Rejected edges
are written back as `NO_EDGE` and cached in a blocked-edge set. Thus

\[
G_t\subseteq G_s\subseteq\widetilde G_s,
\qquad \phi(G_s)=1.
\]

For cycle upper bounds, rejection removes a bond relative to the neural
proposal, which usually relieves rather than creates valence pressure. A final
sanity pass/assertion checks individual upper-bound constraints.

### 6.2 Lower-bound projector after forward edge insertion

The terminal graph is complete and reverse proposals remove edges. Let

\[
D^-=E(G_t)\setminus E(\widetilde G_s)
\]

be the candidate deletions. Starting from $H\leftarrow G_t$, process each
$e\in D^-$ and set

\[
H\leftarrow
\begin{cases}
H-e,&\phi(H-e)=1,\\
H,&\phi(H-e)=0.
\end{cases}
\]

Hence

\[
\widetilde G_s\subseteq G_s\subseteq G_t,
\qquad \phi(G_s)=1.
\]

A rejected deletion forces the sampled topology to keep an edge that the model
wanted absent. With default behavior, the full previous categorical edge
feature is restored. The constraint is therefore guaranteed, but the retained
bond class may be stale and either endpoint may become over-valent.

### 6.3 Conditional positive-class resampling

When `model.resample_blocked_deletions=true`, a blocked deletion retains edge
presence but resamples its class from the reverse posterior conditioned on a
positive edge:

\[
p_\theta(E_{uv}=k\mid E_{uv}\ne0,G_t)
=\frac{p_\theta(E_{uv}=k\mid G_t)}
       {\sum_{j=1}^{d_E-1}p_\theta(E_{uv}=j\mid G_t)},
\quad k=1,\ldots,d_E-1.
\]

The same undirected draw is mirrored at $(u,v)$ and $(v,u)$. If the total
positive mass is zero, conditioning is undefined and the old class is kept.
This rule addresses bond-class staleness; it cannot repair a topology for which
every positive bond order violates valence, nor can it switch which cycle is
preserved through a coordinated multi-edge move.

### 6.4 Guarantees and non-guarantees

Given a feasible terminal state and exact predicate implementation, the
projector guarantees the configured structural predicate at every accepted
reverse state. It does **not** guarantee:

- RDKit molecular validity or atom valence;
- connectivity;
- a chemically meaningful bond-order assignment;
- the globally most likely feasible graph;
- a sample from the data distribution conditioned on the target;
- order independence of the greedy result.

The original ConStruct algorithm describes uniformly random candidate-edge
ordering. This fork obtains candidates from tensor `nonzero` order and processes
them deterministically. The result is maximal with respect to the tested greedy
path, but need not be the unique or globally optimal feasible graph for these
cycle predicates.

## 7. Experimental design represented by the saved matrices

### 7.1 Dataset

The experiments use QM9 with explicit hydrogens removed and at most 9 heavy
atoms. The processed split sizes are:

| Split | Molecules | Share |
|---|---:|---:|
| Train | 97,732 | 74.7% |
| Validation | 20,042 | 15.3% |
| Test | 13,054 | 10.0% |
| Total | 130,828 | 100% |

The train/validation/test cycle distributions are closely aligned. In training,
10.29% of graphs are acyclic, 49.51% have at most one simple cycle, 56.80% have
at most two, 31.76% have maximum cycle length at most four, and 58.72% have
maximum cycle length at most five. Equivalently, 89.71% have at least one cycle,
50.49% have at least two, 79.24% have a cycle of length at least four, and
68.24% have a cycle of length at least five. These base rates matter when
interpreting how invasive each projection target is.

### 7.2 Checkpoints and sampling

| Direction | Transition | Checkpoint | Large-matrix profiles | Samples/profile |
|---|---|---|---:|---:|
| Forward deletion | `absorbing_edges` | epoch 679 | 9 | 10,000 |
| Forward insertion | `edge_insertion` | epoch 1169 | 9 | 10,000 |
| Insertion + blocked-deletion resampling | same epoch 1169 | 9 | 10,000 |

All saved matrices use training/sampling seed 0 and 500 reverse steps
(`faster_sampling=1`). The deletion and original insertion sample sets were
generated on 2026-09-12 and 2026-09-11, respectively, before the 2026-09-14
clean-endpoint correction. The resampling matrix was generated beginning
2026-09-18. Consequently, comparisons between the original insertion and
resampling matrices are useful engineering evidence but are not a perfectly
isolated paired ablation unless sample-generation provenance confirms that only
the resampling flag differs.

### 7.3 Matrix semantics

The generation matrix has rows for count target
$\{\text{disabled},1,2\}$ and length target
$\{\text{disabled},4,5\}$. Each sample set is persisted before evaluation.
Every row is then cross-evaluated against every structural target, producing
constraint heatmaps. Separately, every valid generated set is compared by FCD
with every validation reference cohort, where only the validation side is
filtered by the named target.

For lower bounds, $\ell(G)\ge4$ or $\ell(G)\ge5$ logically implies
$c(G)\ge1$. Therefore the length-only and count-at-least-one-plus-length
profiles are redundant. Their identical rows in the insertion matrices are
expected, not an accidental duplicate result. The corresponding validation
reference cohorts and FCD columns are also identical.

### 7.4 Metrics

- **Structural satisfaction** is computed on every generated graph using the
  exact projector-consistent all-simple-cycle definition.
- **Molecular validity** is the fraction of all generated graphs that RDKit can
  convert/sanitize successfully.
- **Uniqueness** is the fraction of distinct canonical SMILES among valid
  molecules.
- **Novelty** is the fraction of unique valid canonical SMILES absent from the
  canonicalized training set.
- **FCD** is Fréchet ChemNet Distance; lower is better. It is computed only on
  valid canonical SMILES. A method can therefore lose many invalid outputs
  before FCD is computed, so FCD must always be reported beside validity.
- Node-type TV, edge-type TV, charge Wasserstein-1, valency Wasserstein-1,
  disconnectedness, and component counts are also stored per cell.

## 8. Results

### 8.1 Forward edge deletion: upper-bound projection is reliable

The table reproduces the molecular-validity heatmap and the relevant
cross-evaluation columns from
[`metrics.csv`](samples/qm9_no_constraint_edge_absorbing_large/matrix/seed_0/metrics.csv).
Bold entries are the constraints enforced for that row.

| Enforced profile | Validity | $c\le1$ | $c\le2$ | $\ell\le4$ | $\ell\le5$ |
|---|---:|---:|---:|---:|---:|
| None | 98.76% | 39.25% | 47.08% | 27.90% | 50.90% |
| $\ell\le4$ | 99.09% | 75.60% | 91.19% | **100%** | **100%** |
| $\ell\le5$ | 99.05% | 57.16% | 72.08% | 50.68% | **100%** |
| $c\le1$ | 99.69% | **100%** | **100%** | 60.39% | 84.91% |
| $c\le1,\ell\le4$ | 99.72% | **100%** | **100%** | **100%** | **100%** |
| $c\le1,\ell\le5$ | 99.66% | **100%** | **100%** | 70.63% | **100%** |
| $c\le2$ | 99.62% | 76.49% | **100%** | 55.63% | 83.91% |
| $c\le2,\ell\le4$ | 99.69% | 80.86% | **100%** | **100%** | **100%** |
| $c\le2,\ell\le5$ | 99.58% | 75.70% | **100%** | 66.27% | **100%** |

What the table supports:

- All active upper bounds are met in all 10,000 samples per profile.
- Projection does not trade away molecular validity in this matrix. Every
  projected cell is more valid than the 98.76% unprojected cell, by 0.29–0.96
  percentage points.
- A stricter length bound also changes cycle count strongly: enforcing
  $\ell\le4$ raises $c\le1$ prevalence from 39.25% to 75.60%. This is a
  coupled structural effect, not direct count conditioning.
- A count bound changes cycle length even when length is disabled: $c\le1$
  raises $\ell\le4$ prevalence from 27.90% to 60.39%.
- The joint projector behaves as a conjunction and reaches 100% on both active
  targets.

A representative joint cell, $c\le1,\ell\le4$, additionally records 99.72%
validity, 91.18% uniqueness among valid molecules, 92.30% novelty among unique
valid molecules, 0.74% disconnected graphs, 100% planarity, and 4,508.9 seconds
of sampling on one device. Its ordinary test-reference FCD is 4.129; this is
not the same as the target-filtered cross-matrix FCD below.

### 8.2 Forward edge insertion: topology is guaranteed, chemistry degrades

The next table reproduces the original insertion heatmaps and JSON/CSV values.

| Enforced profile | Validity | $c\ge1$ | $c\ge2$ | $\ell\ge4$ | $\ell\ge5$ |
|---|---:|---:|---:|---:|---:|
| None | 98.19% | 86.21% | 45.43% | 75.10% | 63.59% |
| $\ell\ge4$ | 85.51% | **100%** | 54.68% | **100%** | 77.40% |
| $\ell\ge5$ | 80.02% | **100%** | 63.09% | **100%** | **100%** |
| $c\ge1$ | 90.62% | **100%** | 45.44% | 84.45% | 68.78% |
| $c\ge1,\ell\ge4$ | 85.51% | **100%** | 54.68% | **100%** | 77.40% |
| $c\ge1,\ell\ge5$ | 80.02% | **100%** | 63.09% | **100%** | **100%** |
| $c\ge2$ | 63.89% | **100%** | **100%** | 95.81% | 82.90% |
| $c\ge2,\ell\ge4$ | 63.22% | **100%** | **100%** | **100%** | 85.46% |
| $c\ge2,\ell\ge5$ | 61.62% | **100%** | **100%** | **100%** | **100%** |

What the table supports:

- All active lower bounds are met in all 10,000 samples per profile. The hard
  structural projector works as designed.
- Validity decreases monotonically with the main count lower bound: 98.19%
  without projection, 90.62% for $c\ge1$, and 63.89% for $c\ge2$.
- Length lower bounds are also costly: 85.51% at $\ell\ge4$ and 80.02% at
  $\ell\ge5$.
- The strictest joint target loses 36.57 validity points relative to the
  unprojected insertion baseline.
- The lower-bound projector can force the chain onto a thin boundary where a
  topology-critical bond conflicts with the denoiser's preferred chemistry.

The representative $c\ge1,\ell\ge4$ cell records 85.51% validity, 96.82%
uniqueness among valid molecules, 89.02% novelty among unique valid molecules,
1.27% disconnected graphs, 99.79% planarity, and 5,370.6 seconds of sampling.
Its validity-adjusted yield of unique molecules should be considered alongside
its high conditional uniqueness.

### 8.3 Shared-boundary analysis: the asymmetry is not just edge density

Both an at-most-one and at-least-one constraint contain the exactly-one-cycle
boundary:

\[
\{G:c(G)\le1\}\cap\{G:c(G)\ge1\}=\{G:c(G)=1\}.
\]

The saved sample audit reports:

| Generated subset | Graphs | Molecular validity | Mean edges |
|---|---:|---:|---:|
| Deletion diffusion, $c\le1$, exactly one cycle | 9,259 | 99.69% | 8.81 |
| Insertion diffusion, $c\ge1$, exactly one cycle | 5,456 | 84.53% | 8.73 |
| Insertion diffusion, $c\ge1$, more than one cycle | 4,544 | 97.93% | 10.24 |

The first two subsets have almost the same mean edge count, yet differ by
15.16 validity points. Density alone is therefore not an adequate explanation.
Insertion samples with structural slack above the lower bound are much more
valid than those trapped on the boundary. The likely mechanism is directional:

- the upper-bound addition projector removes a rejected proposed edge, usually
  reducing valence load;
- the lower-bound deletion projector restores an edge the model wanted absent,
  potentially increasing valence load or preserving a stale bond order.

Almost every invalid insertion graph in the audit is over-valent. A single
bond-class change can repair 57.5% of invalid graphs under $c\ge1$ and 47.8%
under $c\ge2$. This supports bond-class trapping as an important mechanism,
while leaving room for irreparable topology conflicts.

The conditional distributions should not be equated:

\[
p(G\mid c(G)=1,\text{upper-bound addition path})
\ne
p(G\mid c(G)=1,\text{lower-bound deletion path}).
\]

### 8.4 Conditional resampling is not a complete fix

The same insertion checkpoint was evaluated with conditional positive-class
resampling for blocked deletions:

| Enforced profile | Restore old class | Resample positive class | Change |
|---|---:|---:|---:|
| $\ell\ge4$ | 85.51% | 85.05% | -0.46 pp |
| $\ell\ge5$ | 80.02% | 81.07% | +1.05 pp |
| $c\ge1$ | 90.62% | 90.56% | -0.06 pp |
| $c\ge2$ | 63.89% | 64.87% | +0.98 pp |
| $c\ge2,\ell\ge4$ | 63.22% | 63.91% | +0.69 pp |
| $c\ge2,\ell\ge5$ | 61.62% | 62.50% | +0.88 pp |

Every enforced target remains at 100% satisfaction. The direction of the
validity change is mixed, and the magnitudes are small compared with the gap to
the unprojected baseline. The matched-profile FCD changes are also small. Thus:

- positive-class resampling is technically sound as a way to avoid blindly
  copying the previous class;
- it may help stricter profiles by roughly one point in this seed;
- it does not solve the lower-bound topology/chemistry conflict;
- no superiority claim is justified without matched endpoint code, paired
  proposals where possible, and multiple training and sampling seeds.

### 8.5 Matched-profile FCD

FCD below compares each generated profile with the validation cohort satisfying
the same named target. Lower is better. These values should not be compared
without the accompanying validity because invalid outputs are omitted.

| Profile | Deletion / at-most FCD | Insertion / at-least FCD | Insertion + resampling FCD |
|---|---:|---:|---:|
| None | 1.002 | 0.970 | 0.970 |
| Length 4 | 1.312 | 1.192 | 1.175 |
| Length 5 | 0.996 | 1.369 | 1.376 |
| Count 1 | 1.546 | 1.083 | 1.092 |
| Count 1 + length 4 | 1.380 | 1.192 | 1.175 |
| Count 1 + length 5 | 1.507 | 1.369 | 1.376 |
| Count 2 | 1.173 | 1.920 | 1.920 |
| Count 2 + length 4 | 1.296 | 1.939 | 1.952 |
| Count 2 + length 5 | 1.154 | 2.085 | 2.016 |

Interpretation:

- Projection does not automatically imply poor FCD among the surviving valid
  molecules. Several matched cells remain near 1–1.5.
- Strict insertion count targets have worse matched FCD (about 1.9–2.1) and far
  lower validity, so the degradation is visible in both yield and distribution.
- At-most and at-least columns use different filtered validation cohorts and
  different checkpoints. The table is descriptive; it is not a controlled
  causal ranking of diffusion direction.
- The full 9-by-9 cross-matrices are more informative than their diagonals for
  detecting whether projection shifts samples toward a neighboring target
  cohort.

## 9. A defensible explanation of the directional gap

The following causal chain is consistent with the implementation and saved
diagnostics:

1. Edge deletion forward diffusion produces an empty terminal graph; reverse
   denoising adds bonds.
2. An upper-bound projector handles a violation by replacing a proposed
   positive bond with no-edge. This is chemically conservative with respect to
   valence, although it can disconnect a graph.
3. Edge insertion forward diffusion produces a complete terminal graph;
   reverse denoising removes bonds.
4. A lower-bound projector handles a violation by overriding a proposed
   no-edge with a positive bond. The original rule restores its previous bond
   class even though the model did not choose a positive class on that update.
5. Near the active lower boundary, there may be no single-edge deletion that
   both follows the model and preserves the target. The chain becomes trapped
   with one or more model-opposed bonds.
6. Because the projector sees adjacency only, it cannot change bond order,
   atom type, charge, or another incident bond to recover valence.
7. Structural satisfaction remains perfect while RDKit validity falls.

This explanation is stronger than saying only that “deletion is irreversible.”
Both processes have irreversible boundary states. The meaningful asymmetry is
what the projector does to the neural proposal: **drop an optional positive
edge** for upper bounds versus **force a model-opposed positive edge** for lower
bounds.

## 10. Claims the current evidence supports

Suitable claims, with scope stated explicitly:

1. **Hard topological control.** On 10,000 QM9 samples per seed-0 profile, the
   reverse projectors achieve 100% satisfaction of active simple-cycle count
   and maximum-simple-cycle-length constraints in both directions.
2. **Robust upper-bound behavior.** With the saved edge-deletion checkpoint,
   upper-bound projection preserves or slightly improves molecular validity,
   yielding 99.05–99.72% across constrained profiles.
3. **Topology–chemistry conflict for lower bounds.** With the saved
   edge-insertion checkpoint, stronger lower-bound projection causes a large
   molecular-validity loss despite perfect topological satisfaction, reaching
   61.62% in the strictest profile.
4. **Boundary-specific insertion failure.** Conditional on exactly one final
   cycle, insertion-derived graphs are much less valid than deletion-derived
   graphs at similar mean edge count; insertion graphs with cycle slack are
   much more valid than insertion graphs on the boundary.
5. **Bond-class resampling is insufficient.** Conditional resampling of a
   positive class changes validity by at most about one percentage point in the
   saved matrix and has mixed effects.
6. **A residual endpoint bug disproportionately affected insertion.** The
   paired diagnostic shows that targeting an actually clean final state rescues
   38/1,024 positive-marginal insertion samples and none are harmed, while net
   deletion validity is unchanged.

## 11. Claims the current evidence does not yet support

Avoid these formulations without more experiments:

- “Edge deletion is always better than edge insertion.” Only one dataset and
  one checkpoint per direction are evaluated, and the channels/checkpoints are
  not matched in every respect.
- “Projection samples exactly from QM9 conditioned on the constraint.” The
  projector is a greedy intervention on a generalist model.
- “Resampling improves insertion.” The saved matrix has both gains and losses.
- “The projector alone causes the entire validity gap.” A confirmed historical
  clean-endpoint bug explains a large part of one diagnostic gap; channel
  information, checkpoint quality, and trajectory distribution can also matter.
- “Cycle count means number of rings.” The implementation counts every unique
  simple cycle, which is not the standard chemical SSSR notion.
- “FCD proves high-quality constrained generation” without reporting validity.
  FCD excludes invalid molecules.
- “Statistically robust across seeds.” There is one training seed and one
  matrix sampling seed.

## 12. Limitations and threats to validity

1. **Single seed.** Sampling $10^4$ graphs reduces binomial sampling error,
   but says nothing about checkpoint-to-checkpoint training variance. At least
   three independently trained seeds are needed for mean ± standard deviation.
2. **Historical sampler mismatch.** The deletion and original insertion
   matrices predate the clean-endpoint correction. Both should be regenerated
   from the current commit before final paper tables.
3. **Different checkpoints and epochs.** Directional comparisons combine
   transition design with learned-checkpoint differences.
4. **Greedy/order-dependent projection.** Candidate edge order can select a
   different feasible boundary graph. The implementation order differs from
   the uniformly random ordering described in the original ConStruct paper.
5. **Topology-only constraint.** The projector ignores bond order, valence,
   charges, and chemical sanitization.
6. **Cycle semantics and cost.** All-simple-cycle count can be exponentially
   large and is not interchangeable with chemical ring count.
7. **Feasibility conditioning.** Lower-bound profiles truncate the node-count
   distribution. Some differences may reflect changed size support, so node
   count must be reported and controlled.
8. **Redundant cells.** Lower-bound length 4/5 already implies count at least 1;
   treating the duplicate cells as independent evidence would inflate the
   apparent experiment count.
9. **Validity denominator versus FCD denominator.** Validity uses all generated
   graphs; FCD uses only valid canonical molecules.
10. **Processed-data caveat.** Eleven of the first 4,096 validation tensors in
    the endpoint diagnostic already fail the unchanged RDKit check due to
    carbon valence five. The clean oracle ceiling there is 99.73%.
11. **No connectivity guarantee.** Missing bonds can remain RDKit-valid even
    when graphs are disconnected, which partly makes deletion errors less
    damaging than spurious insertion bonds.

## 13. Experiments needed for a paper-quality result

### Essential reruns

1. Regenerate all three 9-cell matrices from the same current commit after the
   clean-endpoint correction.
2. Train at least three independent seeds for each transition; sample at least
   three seeds per checkpoint or use a clearly nested variance analysis.
3. Report mean ± standard deviation for validity, satisfaction, uniqueness,
   novelty, FCD, node/edge distribution distances, runtime, and blocked-edge
   counts.
4. Fix a shared node-count sequence when comparing methods, in addition to the
   operational feasibility-conditioned result.
5. Save the code commit, config, checkpoint hash, data hash, environment, and
   per-cell sample hash in every manifest.

### Mechanism ablations

Use the same checkpoint, node counts, seed, and—where practical—the same
unprojected reverse proposals:

1. no projection;
2. restore previous class on a blocked deletion;
3. force a single bond;
4. sample a positive class conditionally from the reverse posterior;
5. choose among alternative deletions that preserve the target;
6. allow a local multi-edge swap that preserves cycles and improves valence;
7. add a valence-aware rejection criterion, reported explicitly as an
   additional chemical constraint rather than part of the cycle projector.

Log every blocked event: time, edge, old class, posterior no-edge probability,
positive-class probabilities, endpoint atom types/charges/valences, boundary
status, existence of another permissible deletion, and whether any positive
class is locally valence-valid.

### Boundary-stratified evaluation

For each generated profile, report metrics separately for:

- exact active boundary, e.g. $c(G)=K$ or $\ell(G)=L$;
- slack states, e.g. $c(G)>K$ for a lower bound;
- number of blocked edits per graph;
- graphs with zero, one, or multiple forced edges;
- each node-count stratum.

This directly tests whether validity degrades with projector intervention rather
than merely with the final target property.

### Stronger distribution controls

- Compare a projected generalist with a model trained on the corresponding
  filtered cohort. This distinguishes inference-time controllability from
  specialist modeling.
- Compare the two transitions without projection on identical node counts and
  current endpoint code.
- Include the single-bond insertion endpoint as a controlled channel ablation.
- Report FCD against the unfiltered validation distribution and every filtered
  cohort, but identify the primary endpoint in advance.
- Report the valid-and-target-satisfying yield
  
  \[
  \mathrm{Yield}=\frac{\#\{G:\text{valid}(G)\land\phi(G)\}}
                        {\#\text{generated graphs}},
  \]
  
  which equals validity when projection satisfaction is exactly 100%, but is
  more general and prevents separate metrics from obscuring usable output.

## 14. Suggested paper framing

### Possible central question

> Can ConStruct's monotone constrained diffusion principle be extended from
> edge-deletion-invariant upper-bound properties to edge-addition-invariant
> lower-bound cycle properties without sacrificing molecular validity?

### Possible answer supported by current evidence

> The extension gives exact topological control, but reversing the monotone
> diffusion direction changes the chemical cost of projection. Upper-bound
> projectors reject optional bonds and preserve near-perfect QM9 validity;
> lower-bound projectors may force model-opposed bonds at the constraint
> boundary, producing severe valence failures under strict targets.

### Candidate contribution list

1. A bidirectional monotone formulation for upper- and lower-bound structural
   constraints in discrete graph diffusion.
2. Projectors for all-simple-cycle count and maximum simple-cycle length,
   including conjunctions and feasibility-aware node-count sampling.
3. A systematic generation-by-cross-evaluation matrix separating enforcement
   profiles from post-hoc targets and validation reference cohorts.
4. An empirical characterization of a topology–chemistry asymmetry in molecular
   generation.
5. Diagnostics separating clean-endpoint corruption, topology trapping, and
   bond-class trapping, plus a conditional positive-class resampling ablation.

### Recommended figures and tables

1. **Method figure:** empty-terminal/add-edge/upper-bound branch versus
   complete-terminal/delete-edge/lower-bound branch, showing what happens to a
   rejected edit.
2. **Validity heatmaps:** deletion, insertion, and insertion-resampling on the
   same color scale.
3. **Constraint-satisfaction heatmaps:** one panel for each count/length target.
4. **Matched FCD table:** always adjacent to validity or usable-yield columns.
5. **Boundary plot:** validity versus final distance from the active constraint
   boundary, stratified by number of blocked edits.
6. **Mechanism table:** restore-old-class, force-single, conditional-resample,
   and multi-edge/valence-aware alternatives.
7. **Appendix cross-matrices:** full 9-by-9 FCD matrices and complete provenance.

### Suggested wording for the negative result

> Lower-bound projection was exact with respect to the requested cycle
> predicates but not chemically neutral. As the lower bound became stricter,
> the projector increasingly retained bonds rejected by the denoiser, and
> molecular validity decreased sharply. This separates structural constraint
> satisfaction from molecular validity and shows that a topology-only hard
> projector can require chemically incompatible local edits.

## 15. Reproducibility checklist

- [x] Dataset and split sizes recorded.
- [x] Graph/cycle definitions tied to the processed representation.
- [x] Transition matrices and terminal distributions specified.
- [x] Training loss and major model hyperparameters specified.
- [x] Checkpoint paths, profile counts, sample counts, and seed recorded in
  matrix manifests.
- [x] Immutable generated sample sets separated from replaceable evaluation
  artifacts.
- [x] FCD reference cohort sizes and cache hashes stored.
- [ ] Regenerate all headline matrices after the clean-endpoint fix.
- [ ] Run at least three independent training seeds.
- [ ] Report mean ± standard deviation and paired sampling comparisons.
- [ ] Record current commit/checkpoint/data hashes in all new manifests.
- [ ] Report compute hardware, per-cell runtime, total training time, and cost.
- [ ] Audit train/validation/test duplicates and document QM9 license/source.
- [ ] Predeclare primary metrics and target profiles before final reruns.

## 16. Source-of-truth artifact map

### Method and implementation

- [Noise models and reverse posterior](ConStruct/diffusion/noise_model.py)
- [Denoising model, sampling loop, and node feasibility](ConStruct/diffusion_model_discrete.py)
- [Projector implementation](ConStruct/projector/projector_utils.py)
- [Cycle enumeration](ConStruct/projector/graph_cycles.py)
- [Constraint configuration](ConStruct/projector/constraints.py)
- [Training loss](ConStruct/metrics/train_metrics.py)
- [Molecular metrics](ConStruct/metrics/sampling_molecular_metrics.py)
- [Generalist-model decision](docs/adr/0001-runtime-cycle-constraints.md)
- [Immutable matrix evaluation decision](docs/adr/0002-posthoc-matrix-evaluation.md)
- [Insertion endpoint decision](docs/adr/0003-single-bond-edge-insertion.md)
- [FCD cross-matrix decision](docs/adr/0004-validation-fcd-cross-matrix.md)

### Results

- [Deletion molecular-validity heatmap](samples/qm9_no_constraint_edge_absorbing_large/matrix/seed_0/heatmaps/molecular_validity.png)
- [Insertion molecular-validity heatmap](samples/qm9_no_constraint_edge_addition_large/matrix/seed_0/heatmaps/molecular_validity.png)
- [Insertion-resampling molecular-validity heatmap](samples/qm9_no_constraint_edge_addition_large_resampling/matrix/seed_0/heatmaps/molecular_validity.png)
- [QM9 training cycle heatmap](qm9_train_cycle_heatmap.png)

- [Deletion matrix](samples/qm9_no_constraint_edge_absorbing_large/matrix/seed_0/metrics.csv)
- [Original insertion matrix](samples/qm9_no_constraint_edge_addition_large/matrix/seed_0/metrics.csv)
- [Insertion-resampling matrix](samples/qm9_no_constraint_edge_addition_large_resampling/matrix/seed_0/metrics.csv)
- [Deletion matrix manifest](samples/qm9_no_constraint_edge_absorbing_large/matrix/seed_0/manifest.json)
- [Insertion matrix manifest](samples/qm9_no_constraint_edge_addition_large/matrix/seed_0/manifest.json)
- [Resampling matrix manifest](samples/qm9_no_constraint_edge_addition_large_resampling/matrix/seed_0/manifest.json)
- [QM9 train cycle heatmap data](qm9_train_cycle_heatmap.json)
- [Processed-QM9 cycle audit](diagnostics/qm9_cycles/audit.json)
- [Shared-boundary validity analysis](EDGE_INSERTION_BOUNDARY_VALIDITY.md)
- [Clean-endpoint investigation](diagnostics/edge_boundary/README.md)
- [Clean-endpoint counts](diagnostics/edge_boundary/results.json)
- [Trajectory diagnostics](diagnostics/trajectory_gap/summary.md)

## 17. Compact takeaway for an abstract or conclusion

The project's strongest current finding is a directional asymmetry in hard
constrained molecular graph diffusion. Both monotone processes can guarantee
cycle constraints, but the intervention required at the boundary matters.
Rejecting an added edge under an upper bound is chemically forgiving; rejecting
a deleted edge under a lower bound can force a bond opposed by the denoiser and
violate valence. On the saved QM9 matrices this distinction separates
near-perfect validity under deletion diffusion from losses of up to 36.57
percentage points under insertion diffusion. The result motivates constrained
samplers that reason jointly about topology and categorical bond chemistry,
rather than treating a structurally feasible adjacency as sufficient.
