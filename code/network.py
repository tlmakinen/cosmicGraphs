
import jax
import jax.numpy as jnp
import jraph
from flax import linen as nn
from flax import optim
from typing import Any, Callable, Sequence, Optional

import optax

# custom scaling function for large NN inputs
class AsinhLayer(nn.Module):
    bias_init: Callable = nn.initializers.zeros
    a_init: Callable = nn.initializers.ones
    b_init: Callable = nn.initializers.ones
    c_init: Callable = nn.initializers.zeros
    d_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, inputs):

        a = self.param('a', self.a_init, (1,)) #*(1./36)
        b = self.param('b', self.b_init, (1,))
        c = self.param('c', self.c_init, (1,))
        d = self.param('d', self.d_init, (1,))

        y = a*jnp.arcsinh(b*inputs + c) + d
        return y

# define activation
act = nn.gelu

# fully-connected network
class ExplicitMLP(nn.Module):
  """A flax MLP."""
  features: Sequence[int]

  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
      x = lyr(x)
      if i != len(self.features) - 1:
        x = act(x)
    return x

# adapt for jraph library -- need wrappers for linen Modules
def make_embed_fn(latent_size):
  def embed(inputs):
    inputs = AsinhLayer()(inputs)
    return nn.Dense(latent_size)(inputs)
  return embed


def make_mlp(features):
  @jraph.concatenated_args
  def update_fn(inputs):
    return ExplicitMLP(features)(inputs)
  return update_fn


# custom mean function for padded inputs
def custom_segment_mean(
                 n_data: int,
                 data: jnp.ndarray,
                 segment_ids: jnp.ndarray,
                 num_segments: Optional[int] = None,
                 indices_are_sorted: bool = False,
                 unique_indices: bool = False):
  """Returns mean for each segment.
  Args:
    n_data: the number of data we want to take the mean of
    data: the values which are averaged segment-wise.
    segment_ids: indices for the segments.
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  """
  denom = n_data
  nominator = jraph.segment_sum(
      data,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  denominator = jraph.segment_sum(
      jnp.ones_like(data),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)
  return nominator / jnp.maximum(denom,
                                 jnp.ones(shape=[], dtype=denominator.dtype))

# custom variance function for padded inputs
def custom_segment_variance(n_data: int,
                     data: jnp.ndarray,
                     segment_ids: jnp.ndarray,
                     num_segments: Optional[int] = None,
                     indices_are_sorted: bool = False,
                     unique_indices: bool = False):
  """Returns the variance for each segment.
  Args:
    n_data: the number of data we want to take the variance of
    data: values whose variance will be calculated segment-wise.
    segment_ids: indices for segments
    num_segments: total number of segments.
    indices_are_sorted: whether ``segment_ids`` is known to be sorted.
    unique_indices: whether ``segment_ids`` is known to be free of duplicates.
  Returns:
    num_segments size array containing the variance of each segment.
  """
  means = custom_segment_mean(
      n_data,
      data,
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)[segment_ids]
  counts = jraph.segment_sum(
      jnp.ones_like(data),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices)

  counts = n_data*jnp.ones_like(counts)
  counts = jnp.maximum(counts, jnp.ones_like(counts))

  variances = jraph.segment_sum(
      jnp.power(data - means, 2),
      segment_ids,
      num_segments,
      indices_are_sorted=indices_are_sorted,
      unique_indices=unique_indices) / counts
  return variances

# custom aggregation function
def custom_aggregation(n_data: int,
                     data: jnp.ndarray,
                     segment_ids: jnp.ndarray,
                     num_segments: Optional[int] = None,
                     indices_are_sorted: bool = False,
                     unique_indices: bool = False):
    """Returns the variance for each segment.
      Args:
        n_data: the number of data we want to take the variance of
        data: values whose variance will be calculated segment-wise.
        segment_ids: indices for segments
        num_segments: total number of segments.
        indices_are_sorted: whether ``segment_ids`` is known to be sorted.
        unique_indices: whether ``segment_ids`` is known to be free of duplicates.
      Returns:
        (num_segments, 4) size array containing the [sum, mean, variance, max]
        of input data.
    """


    variance = custom_segment_variance(n_data=n_data,
                     data=data,
                     segment_ids=segment_ids,
                     num_segments=num_segments,
                     indices_are_sorted=indices_are_sorted,
                     unique_indices=unique_indices)

    mean = custom_segment_mean(n_data=n_data,
                     data=data,
                     segment_ids=segment_ids,
                     num_segments=num_segments,
                     indices_are_sorted=indices_are_sorted,
                     unique_indices=unique_indices)

    _max = jraph.segment_max(
                     data=data,
                     segment_ids=segment_ids,
                     num_segments=num_segments,
                     indices_are_sorted=indices_are_sorted,
                     unique_indices=unique_indices)

    _sum = jraph.segment_sum(
                     data=data,
                     segment_ids=segment_ids,
                     num_segments=num_segments,
                     indices_are_sorted=indices_are_sorted,
                     unique_indices=unique_indices)

    return jnp.concatenate([_sum, mean, variance, _max], axis=-1)


class flaxGraphNetwork(nn.Module):
  """A flax GraphNetwork."""
  mlp_features: Sequence[int]
  latent_size: int
  decorate_nodes: bool=False
  remove_vel: bool=False
  num_nets: int=1
  remove_edges: bool=False

  @nn.compact
  def __call__(self, graph):

    _nodes = graph.nodes

    if not self.decorate_nodes:
        # the indicator functions are here represented by the edge
        # sender and receiver indexes, so we are free to set all nodes
        # to zero.
        _nodes = _nodes.at[:, 0:4].set(0.)

    # in this study we don't consider velocity
    if self.remove_vel:
        _nodes = _nodes.at[:, 1:4].set(0.)

    # whether or not to remove edge labels. here always False.
    if self.remove_edges:
        graph = graph._replace(edges=jnp.zeros(graph.edges.shape))

    # replaces graph features with desired masked elements.
    # add N^v and N^e as global properties
    graph = graph._replace(
                           globals=jnp.vstack([jnp.arcsinh(graph.n_node), jnp.arcsinh(graph.n_edge)]).reshape(1,-1),
                           senders=graph.senders.astype(int),
                           receivers=graph.senders.astype(int),
                           n_node = graph.n_node.reshape(-1,1),
                           nodes=_nodes)

    embedder = jraph.GraphMapFeatures(
        embed_node_fn=make_embed_fn(self.latent_size),
        embed_edge_fn=make_embed_fn(self.latent_size),
        embed_global_fn=make_embed_fn(self.latent_size))

    # rho aggregation functions
    aggregate_nodes_for_globals_fn = lambda d,s,n: jnp.arcsinh(custom_aggregation(jnp.squeeze(graph.n_node), d,s,n))
    aggregate_edges_for_nodes_fn = jraph.segment_sum #lambda d,s,n: jnp.arcsinh(jraph.segment_sum(d,s,n))
    aggregate_edges_for_globals_fn = lambda d,s,n: jnp.arcsinh(custom_aggregation(jnp.squeeze(graph.n_edge), d,s,n))

    update_node_fn = make_mlp(self.mlp_features)
    update_edge_fn = make_mlp(self.mlp_features)

    # first embed the graph features to a higher dimensionality
    graph = embedder(graph)

    # then pass through N^int interaction GNN blocks
    for i in range(self.num_nets):
        if i == self.num_nets-1:
            feats = self.mlp_features + (n_params,)
        else:
            feats = self.mlp_features

        net = jraph.GraphNetwork(
            update_node_fn=make_mlp(self.mlp_features),
            update_edge_fn=make_mlp(self.mlp_features),
            aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
            aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
            aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
            update_global_fn=make_mlp(feats))

        graph = net(graph)

    # return global properties (gIMNN summaries)
    return graph.globals.reshape(-1)
