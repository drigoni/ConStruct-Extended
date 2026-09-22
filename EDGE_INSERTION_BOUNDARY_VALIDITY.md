# Edge-insertion constraints and molecular validity

## Question

The `absorbing_edges` sampler keeps molecular validity near 99% when an upper-bound cycle constraint is active. The `edge_insertion` sampler loses validity under lower-bound constraints:

| Sampling profile | Molecular validity |
| --- | ---: |
| Edge insertion, unconstrained | 98.19% |
| Edge insertion, at least 1 cycle | 90.62% |
| Edge insertion, at least 2 cycles | 63.89% |
| Absorbing edges, unconstrained | 98.76% |
| Absorbing edges, at most 1 cycle | 99.69% |
| Absorbing edges, at most 2 cycles | 99.62% |

These values come from the 10,000-graph `seed_0` matrices in `samples/qm9_no_constraint_edge_addition_large` and `samples/qm9_no_constraint_edge_absorbing_large`.

The first explanation was that deletion gets stuck at a lower-bound constraint. That explanation is incomplete. Addition also gets stuck at an upper-bound boundary. Once an edge creates the first cycle in a graph constrained to at most one cycle, the sampler cannot later delete that cycle. Both processes have irreversible boundary states.

## The shared boundary

Graphs with exactly one cycle lie in the intersection of the two constraint sets:

\[
\{G : c(G) \leq 1\} \cap \{G : c(G) \geq 1\}
= \{G : c(G)=1\}.
\]

A CPU audit of the generated batches compared validity inside this shared slice:

| Generated subset | Graphs | Molecular validity | Mean edges |
| --- | ---: | ---: | ---: |
| `absorbing_edges`, at most 1 cycle, exactly-one-cycle subset | 9,259 | 99.69% | 8.81 |
| `edge_insertion`, at least 1 cycle, exactly-one-cycle subset | 5,456 | 84.53% | 8.73 |
| `edge_insertion`, at least 1 cycle, more-than-one-cycle subset | 4,544 | 97.93% | 10.24 |

The exactly-one-cycle graphs have almost the same mean edge count. Density alone does not explain the validity gap. The insertion samples become much more valid when they have more than one cycle and therefore have slack above the lower bound.

Conditioning on the same final property does not make the path distributions equal:

\[
p(G \mid c(G)=1,\text{ upper-projected addition})
\neq
p(G \mid c(G)=1,\text{ lower-projected deletion}).
\]

## How each projector reaches the boundary

Let \(\widetilde G_s\) denote the graph proposed by the reverse kernel.

The at-most projector starts from an empty graph. It rejects an addition when that addition would exceed the cycle limit. Its output is a subgraph of the proposal:

\[
G_s \subseteq \widetilde G_s.
\]

A cycle that survives in the final graph was assembled from edges proposed by the model. The at-most-one constraint never forces the sampler to create a cycle. It can return an acyclic graph; 7.41% of the saved at-most-one samples do so.

The at-least projector starts from a complete graph. It rejects a deletion when that deletion would cross below the cycle limit. Its output is a supergraph of the proposal:

\[
G_s \supseteq \widetilde G_s.
\]

At the one-cycle boundary, the model can propose deleting an edge while the projector restores it. The surviving cycle may therefore contain an edge that the model wanted absent. With more than one cycle, the sampler can remove a bad edge and preserve another cycle. This matches the sharp validity difference between the exactly-one and more-than-one insertion subsets.

The two projectors apply opposite residuals to the model proposal. The upper-bound projector replaces a rejected positive edge with no edge. The lower-bound projector replaces a rejected no-edge prediction with a positive edge. Those operations have different chemical costs. Removing a bond usually relieves valence pressure. Restoring a bond can make either endpoint over-valent.

## Bond classes are not projected

The projector reduces the edge tensor to binary adjacency before finding candidate mutations. It sees edge creation and deletion, but it does not see changes among single, double, triple, and aromatic classes.

For `edge_insertion`, a blocked deletion restores the complete edge feature from the previous state. If the model proposed no edge, it did not select a chemically compatible positive class for that update. The projector keeps the old class anyway. At the final step, this can leave a stale bond order on a constraint-critical edge.

The batch audit found that almost every invalid insertion graph was over-valent. A single bond-class change repaired 57.5% of invalid graphs under the at-least-one constraint and 47.8% under at least two cycles. The structural projector cannot make either repair because the adjacency does not change.

## Proposed sampler change

A blocked deletion should not count as an ordinary denoising update. Copying the previous edge class silently substitutes a stale class for the model's no-edge prediction.

One candidate rule is:

1. Sample the next edge state from the reverse posterior.
2. If the sampled state is positive, keep the normal update.
3. If the sampled state is no-edge, test the deletion against the lower-bound constraint.
4. If deletion is allowed, keep no-edge.
5. If deletion is blocked, keep the adjacency but choose a positive bond class from the model distribution conditioned on edge presence:

   \[
   p(E=k \mid E\neq 0, z_t), \qquad k\in\{1,2,3,4\}.
   \]

This requires passing the predicted or posterior edge-class probabilities into the projector. The current projector receives only the sampled `z_s`, so it cannot distinguish a confident no-edge prediction from an uncertain one or resample a positive class from the same prediction.

The conditional-class rule addresses the stale-class failure. It cannot repair every topology failure. If any positive class on the protected edge makes an atom over-valent, the lower-bound cycle constraint and chemical validity are locally incompatible. The sampler then needs a larger move, such as preserving a different cycle, changing another incident bond, or jointly proposing several edges.

A simpler control would force blocked deletions to become single bonds. That should reduce valence errors, but it imposes a hand-written class bias. It is useful as an ablation against conditional positive-class sampling, not as the default design without evidence.

## Tests needed

The next experiment should separate topology trapping from bond-class trapping.

For every blocked deletion, log:

- the timestep and graph index;
- the deleted edge and its previous class;
- the model's probability of no-edge;
- the model's probabilities over positive classes;
- endpoint atom types, charges, and valences;
- whether another deletion could preserve the constraint;
- whether any positive class would preserve chemical valence.

Compare three lower-bound projectors with the same checkpoint, seed, node counts, and reverse proposals:

1. restore the previous class, which is the current behavior;
2. restore a single bond;
3. sample from the model distribution conditioned on a positive class.

Report molecular validity separately for graphs at the active boundary and graphs with slack. Also report constraint satisfaction, bond-type distribution, FCD, and the number of forced edges. A validity gain paired with distorted bond types or worse FCD would show that the change repairs valence by biasing the generated distribution.

The saved matrices were generated on September 11 and 12, 2026, before the September 14 clean-endpoint correction in `ConStruct/diffusion/noise_model.py`. New samples are required before assigning the full gap to the projector. The present audit uses one trained seed, so it supports the mechanism but does not establish its stability across training runs.

## Implemented switch

Set `model.resample_blocked_deletions=true` to enable conditional positive-class
sampling for blocked deletions under `edge_insertion` or `edge_insertion_single`.
The default, `false`, restores the previous class for compatibility with existing
runs. Individual and joint lower-bound projectors support the switch.

The conditional distribution uses the reverse posterior that generated the
proposal, removes the no-edge class, and renormalizes the positive classes.
Each undirected edge receives one draw mirrored to both tensor entries. This
also applies to subsequent deletion proposals for already protected edges.
Allowed deletions and positive-class proposals follow their normal updates.
If positive posterior mass is exactly zero, the previous class is retained
because the conditional distribution is undefined. The flag is recorded in
sampling metrics metadata. This switch does not enforce chemical validity.
