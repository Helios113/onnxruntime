# Training Roadmap (future directions)

## Gradients through control flow (`Loop`, `If`, `Scan`)

**Status:** not supported.

ORT training has no gradient builders for `Loop`, `If` or `Scan`. Building the gradient graph for a model with
one of these nodes on a path between the loss and a trainable input/parameter fails in
`GradientBuilderRegistry::CreateGradientDefs`
(`orttraining/orttraining/core/graph/gradient_builder_registry.cc`) with
"The gradient builder has not been registered: <op> for node <name>".

Nothing in the gradient builders or `GradientGraphBuilder` reads subgraph attributes (`body`, `then_branch`,
`else_branch`), so this needs new infrastructure, not just a registration.

### Possible approach
- `If`: build a gradient node whose `then_branch`/`else_branch` are the gradient subgraphs of the original branches.
  Both branches must produce gradients for the same set of inputs (zeros for unused ones).
- `Scan`: generate a reverse scan over the gradient of the body, stashing the per-iteration forward activations.
- `Loop`: needs the trip count and per-iteration activations stashed in the forward pass, then a reverse loop
  over the gradient body. Dynamic trip counts make stashing the hardest part.
- Gradient construction for subgraphs, including outer-scope values captured by the body.

### Interim workarounds
- Unroll loops at export time (fixed trip count).
- Replace `If` with `Where` and compute both branches.
- Register `NO_GRADIENT` for control-flow nodes that are not on the gradient path (e.g. only compute indices/masks).
