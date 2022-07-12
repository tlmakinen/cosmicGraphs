import jax.random as jrnd
from functools import partial
import scipy.spatial as SS

import numpy as onp

from functools import partial
import jax
import jax.random as jrnd
import jax.numpy as jnp
import jraph
from scipy.sparse import csgraph
import numpy as np
import os, sys
from struct import unpack

import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cloudpickle as pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def get_freqs(shape, L, fourier_b=1.0):
    # set frequencies for jittable fft calculation

    axes = list(range(len(shape)))
    _N = np.array([shape[axis] for axis in axes])

    # Get the box volume if given the real-space box volume
    _L = np.array([L] * len(axes))
    dx = _L / _N
    Lk = 2 * np.pi / (dx * fourier_b)

    Lk = np.array(Lk)
    left_edge = _set_left_edge(None, axes, Lk)

    V = np.product(Lk)
    dk = np.array(Lk) / np.array(_N)

    _myfreq = lambda n,d: fftfreq(n, d=d, b=fourier_b)
    freq = jax.tree_util.tree_map(_myfreq, list(_N), list(dk))
    return freq, axes, left_edge


##### GET NODES FROM FIELD #####
def get_nodes_from_field(key,
                         deltax,
                         pb,
                         pad_nodes_to=500,
                         nbar=0.000025,
                         randomise_in_cell=True,
                         min_at_zero=True,
                         get_context=True,
                         window=2):
    """
    function for grabbing mock halos from density field
    """

    key,rng = jax.random.split(key)

    N = pb.N
    n = (deltax + 1) * pb.dx ** pb.dim * nbar
    n_per_cell = jax.random.poisson(key, n.flatten(), shape=n.flatten().shape)

    mask = (n_per_cell > 0)
    num_nodes = jnp.sum(mask).astype(int)

    # make field positive
    deltax += 1.0
    density = deltax.flatten().repeat(
                      n_per_cell.flatten(), axis=0,
                      total_repeat_length=pad_nodes_to
                  )
    # remove dumb repeat padding with zeros
    #TODO: is there a cleaner way to do this ?
    density = jnp.where(density == density[-1], 0., density)
    # Get all source positions
    args = [pb.x] * pb.dim
    X = jnp.meshgrid(*args)
    tracer_positions = jnp.array([x.flatten() for x in X]).T
    tracer_positions = tracer_positions.repeat(
                              n_per_cell.flatten(), axis=0,
                              total_repeat_length=pad_nodes_to)

    if randomise_in_cell:
        noise = (jax.random.uniform(key, shape=(tracer_positions.shape[0],
                                   pb.dim)) * pb.dx)

                                  #.repeat(jnp.sum(n_per_cell), axis=0,
                                  #     total_repeat_length=pad_nodes_to)

        noise = jnp.where((density == 0.)[:, jnp.newaxis], 0., noise)

        tracer_positions += noise


    if min_at_zero:
        tracer_positions += pb.boxlength / 2.0

    # pad with zeros
    val = tracer_positions[:, 0][-1]
    tracer_positions = jnp.where((tracer_positions[:, 0] == val)[:, jnp.newaxis],
                                            0., tracer_positions)

    if get_context:

            # Get all source indexes
        args = [jnp.arange(pb.N)] * pb.dim
        X = jnp.meshgrid(*args)
        indexes = jnp.array([x.flatten() for x in X]).T
        indexes = indexes.repeat(
                                 n_per_cell.flatten(), axis=0,
                                 total_repeat_length=pad_nodes_to)

        neighbor_sums = jnp.zeros(indexes.shape[0]) #[]
        idx = jnp.arange(deltax.flatten().shape[0]).reshape(N,N,N)

        w = window
        i_end = N-1
        j_end = i_end
        k_end = i_end

        for l,coord_pair in enumerate(indexes):
            i,j,k = coord_pair

            pad = 2
            w = 3

            neighborhood = jax.lax.dynamic_slice(jnp.pad(deltax, pad_width=pad),
                                  (i+pad -w//2, j+pad-w//2, k+pad-w//2), (w,w,w))
            neighbor_sums = neighbor_sums.at[l].set(jnp.sum(neighborhood))#.append(jnp.sum(neighborhood))

        #neighbor_sums = jnp.array(neighbor_sums)
        neighbor_sums = jnp.where(density == 0., 0., neighbor_sums)

        return tracer_positions, density, neighbor_sums

    else:
        return tracer_positions, density, jnp.zeros(density.shape)

#####################


def get_distances(X):
    nx = X.shape[0]
    return (X[:, None, :] - X[None, :, :])[jnp.tril_indices(nx, k=-1)]


def get_receivers_senders(nx, dists, connect_radius=0.15):
    '''connect nodes within `connect_radius` units'''

    senders,receivers = jnp.tril_indices(nx, k=-1)
    dists = dists[jnp.tril_indices(nx, k=-1)]
    mask = dists < connect_radius
    # pad dummy s,r with n_node
    senders = jnp.where(mask > 0, senders, nx)
    receivers = jnp.where(mask > 0, receivers, nx)
    dists = jnp.where(mask > 0, dists, 0.)
    return senders, receivers, dists

def l2norm_einsum(X, eps=1e-9):
    """calculaute eucl distance with einsum"""
    a_min_b = X[:, None, :] - X[None, :, :]
    norm_sq = jnp.einsum("ijk,ijk->ij", a_min_b, a_min_b)
    return jnp.where(norm_sq < eps, 0, jnp.sqrt(norm_sq))

def get_r2(X):
    """calculate euclidean distance from positional information"""
    nx = X.shape[0]
    #alldists = l2norm(X[:, None, :] - X[None, :, :])
    alldists = l2norm_einsum(X)
    return alldists #[jnp.tril_indices(nx, k=-1)]


##### GET EDGES #####
def edge_builder(pos, r_connect, n_node=None, invert_edges=True,
                     boxsize=1.0001, leafsize=16):

    if n_node is not None:
        pos = pos[:n_node]

    else:
      n_node = pos.shape[0]

    r_connect = r_connect #simulator_args["connect_radius"] / simulator_args["L"]

    #print(r_connect)

    # kd_tree = SS.KDTree(pos, leafsize=leafsize, boxsize=boxsize)
    # edge_index = onp.array(list((kd_tree.query_pairs(r=r_connect)))).T
    # _senders,_receivers = edge_index

    # mask out halos with distances < connect_radius
    dists = get_r2(pos)

    _receivers, _senders, dists = get_receivers_senders(n_node,
                                                        dists,
                                                        connect_radius=r_connect)

    diff = pos[_senders] - pos[_receivers]

    # Take into account periodic boundary conditions, correcting the distances
    # for i, pos_i in enumerate(diff):
    #     for j, coord in enumerate(pos_i):
    #         if coord > r_connect:
    #             diff[i,j] -= 1.  # Boxsize normalize to 1
    #         elif -coord > r_connect:
    #             diff[i,j] += 1.  # Boxsize normalize to 1


    num_pairs = dists.shape[0] #edge_index.shape[-1]
    row = _senders
    col = _receivers

    # Distance
    dist = dists #jnp.linalg.norm(diff, axis=1)

    # Centroid of galaxy catalogue
    centroid = jnp.mean(pos,axis=0)

    # Unit vectors of node, neighbor and difference vector
    unitrow = (pos[row]-centroid)/jnp.linalg.norm((pos[row]-centroid), axis=1).reshape(-1,1)
    unitcol = (pos[col]-centroid)/jnp.linalg.norm((pos[col]-centroid), axis=1).reshape(-1,1)

    unitdiff = jnp.where((dist.reshape(-1,1) > 0.), diff/dist.reshape(-1,1), 1.)

    # Dot products between unit vectors
    #TODO: einsum this
    # cos1 = jnp.array([jnp.dot(unitrow[i,:].T,unitcol[i,:]) for i in range(num_pairs)])
    # cos2 = jnp.array([jnp.dot(unitrow[i,:].T,unitdiff[i,:]) for i in range(num_pairs)])
    cos1 = jnp.einsum('ij,ij->i', unitrow, unitcol)
    cos2 = jnp.einsum('ij,ij->i', unitrow, unitdiff)

    # mask out nans
    cos1 = jnp.where(dist == 0., 0., cos1)
    cos2 = jnp.where(dist == 0., 0., cos2)

    if invert_edges:
        # flip the distance
        dist = jnp.where((dist > 0.), 1. / (dist*r_connect*100.), dist)
        # sort edges from biggest to smallest
        idx = jnp.argsort(dist)[::-1]
        dist = jnp.sort(dist)[::-1]

    else:
        # Normalize distance by linking radius
        dist /= r_connect
        # pad with large dummy edge
        mask = (dist > 0.)
        fillval = 100.
        dist = jnp.where(mask < 1, fillval, dist)

        # sort edges from SMALLEST to BIGGEST
        idx = jnp.argsort(dist) #[::-1]
        dist = jnp.sort(dist) #[::-1]

        # replace all dummy distances with zeros again
        dist = jnp.where(dist == fillval, 0., dist)



    cos1 = cos1[idx]
    cos2 = cos2[idx]

    _senders = _senders[idx]
    _receivers = _receivers[idx]

    edge_attr = jnp.concatenate([dist.reshape(-1,1), cos1.reshape(-1,1), cos2.reshape(-1,1)], axis=1)

    return edge_attr, jnp.array(_senders), jnp.array(_receivers)


##### GET CATALOG BUILDER #####

def halo_catalog_builder(key, θ, deltax, lnpb, simulator_args):
    #A,B = θ

    pad_nodes_to = simulator_args['pad_nodes_to']
    pad_edges_to = simulator_args['pad_edges_to']
    nbar = simulator_args["nbar"] # 0.000025
    dim = simulator_args['dim']
    get_context = simulator_args["get_context"]
    window = simulator_args["window"]

    key,rng = jax.random.split(key)

    x,masses,neighbor_density = get_nodes_from_field(
                     key,
                     deltax,
                     lnpb,
                     randomise_in_cell=True,
                     get_context=get_context,
                     pad_nodes_to=pad_nodes_to,
                     nbar=simulator_args["nbar"])

    # set positions to size-1 box:
    x /= simulator_args["L"]



    # nodes for graphs
    #node_attr = jnp.concatenate([masses.reshape(-1,1), neighbor_density.reshape(-1,1)], axis=-1)

    # full catalog (mass, context, positions)
    catalog = jnp.concatenate([masses.reshape(-1,1), neighbor_density.reshape(-1,1), x], axis=-1)

    return catalog

##### GET GRAPH PADDING #####

def padded_graph_builder(catalog, θ, simulator_args, r_connect=None, n_node=None):

    pad_nodes_to = simulator_args['pad_nodes_to']
    pad_edges_to = simulator_args['pad_edges_to']
    nbar = simulator_args["nbar"] # 0.000025
    dim = simulator_args['dim']
    get_context = simulator_args["get_context"]
    window = simulator_args["window"]
    include_pos = simulator_args["include_pos"]
    node_features = simulator_args["node_features"]

    #if r_connect is None:
        #r_connect = simulator_args["connect_radius"] / simulator_args["L"]

    #else:
        #r_connect /= simulator_args["L"]

    # get empty arrays

    if include_pos:
        nodes = jnp.zeros((pad_nodes_to, node_features))
        edges = jnp.zeros((pad_edges_to, 1)) # dist

    else:
        nodes = jnp.zeros((pad_nodes_to, 4)) # density, neighborhood info
        edges = jnp.zeros((pad_edges_to, 3)) # dist, angle, angle

    senders = (jnp.ones((pad_edges_to), dtype=int)*pad_nodes_to).astype(int)
    receivers = (jnp.ones((pad_edges_to), dtype=int)*pad_nodes_to).astype(int)


    # dealing with the padded catalog
    # set the dummy nodes' positions to way outside the box
    catalog = jnp.where((catalog[:, 0] == 0.)[:, jnp.newaxis],
                        jnp.array([0., 0., 0., 0., 100., 100., 100.]),
                        catalog)

    # unpack node attributes from catalog
    pos = catalog[:, 4:]

    # reset dummy positions to zero after passing to pos
    catalog = jnp.where((catalog[:, 0] > 0.)[:, jnp.newaxis], catalog, 0.)

    if include_pos:
        node_attr = catalog[:, :]

    else:
        node_attr = catalog[:, :4]

    n_node = jnp.sum(node_attr[:, 0] > 0.)

    # GET EDGE INFORMATION
    edge_attr, _s, _r = edge_builder(jnp.array(pos), r_connect=r_connect,
                                     invert_edges=simulator_args["invert_edges"],
                                     boxsize=1.0001)

    if include_pos:
        edge_attr = edge_attr[:, :1]

    n_edge = jnp.sum(edge_attr[:, 0] > 0.)

    # fill in jax arrays

    # edge information
    edges = edges.at[:pad_edges_to, :].set(edge_attr[:pad_edges_to, :])
    senders = senders.at[:pad_edges_to].set(_s[:pad_edges_to])
    receivers = receivers.at[:pad_edges_to].set(_r[:pad_edges_to])

    # add in node information
    nodes = nodes.at[:pad_nodes_to, :].set(node_attr[:pad_nodes_to])

    if simulator_args["squeeze"]:

        graph = jraph.GraphsTuple(nodes=jnp.squeeze(nodes), edges=jnp.squeeze(edges),
                              senders=jnp.squeeze(senders), receivers=jnp.squeeze(receivers),
                              n_node=jnp.array([n_node]), n_edge=jnp.array([n_edge]), globals=jnp.array([θ])
                              )
        return graph

    else:
        graph = jraph.GraphsTuple(nodes=nodes, edges=edges,
                                senders=senders, receivers=receivers,
                                n_node=jnp.array([n_node]), n_edge=jnp.array([n_edge]), globals=jnp.array([θ])
                                )
        return graph


##### WRAPPER FOR GETTING A GRAPH DIRECTLY FROM DENSITY FIELD #####

def graphs_from_density_field(key, θ, deltax, lnpb, simulator_args):
    """assemble a single, batchable graph from a density field"""

    catalog = halo_catalog_builder(key, θ_fid, deltax, lnpb, simulator_args=simulator_args)
    graph = padded_graph_builder(catalog, θ_fid, simulator_args=simulator_args)

    return graph


####################### VISUALIZE GRAPH #######################

def plot_graph(graph,
                ax=None,
                nodesize=45,
                nodealpha=0.5,
                edgewidth=1.5,
                edgealpha=0.5,
                labelaxes=False,
                removeticks=False,
                removebox=False):
    """visualize jraph graph using Networkx"""

    send_receive = [(int(graph.senders[l]), int(graph.receivers[l])) for l in range(len(graph.receivers))]

    G = nx.Graph()
    G.add_nodes_from(list(np.arange(graph.nodes[:, :1].shape[0])))
    G.add_edges_from(send_receive)


    # 3d spring layout
    pos = graph.nodes[:, 1:4] #X

    masses = graph.nodes[:, 0]


    # Extract node and edge positions from the layout
    node_xyz = np.array([pos[v] for v in sorted(G)])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in G.edges()])

    if ax is None:
            # Create the 3D figure
        fig = plt.figure(figsize=(7,4))
        ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    sc = ax.scatter(*node_xyz.T, s=nodesize, ec='w',
                    c=masses, cmap='gist_gray', alpha=nodealpha)



    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray", lw=edgewidth, alpha=edgealpha)


    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        if removeticks:
            for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
                dim.set_ticks([])

        if removebox:
            for key, spine in ax.spines.items():
                spine.set_visible(False)
            ax.axis('off')
        if labelaxes:
            # Set axes labels
            ax.set_xlabel(r"$x\ \rm [Mpc/h]$", fontsize=15)
            ax.set_ylabel(r"$y$", fontsize=15)
            ax.set_zlabel(r"$z$", fontsize=15)


    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')


    ax.view_init(azim=15, elev=20)
    _format_axes(ax)
    #plt.show()

    return ax

def load_single_unpadded_graph(fname, idx, masscut, r_connect):
  _graph = load_obj(fname)
  catalog = _graph.nodes[idx] # take first simulation
  catalog = catalog[catalog[:, 0] > masscut] # remove padding, with masscut

  pos = catalog[:, 4:]

  catalog = jnp.concatenate([catalog[:, :1], catalog[:, 4:]], axis=-1)

  edge_attr, _s, _r = edge_builder(jnp.array(pos), r_connect=r_connect,
                                  invert_edges=simulator_args["invert_edges"],
                                  boxsize=1.0001)

  edgemask = edge_attr[:, 0] > 0
  edge_attr = edge_attr[edgemask, 0]
  _s = _s[edgemask]
  _r = _r[edgemask]

  return jraph.GraphsTuple(nodes=catalog,
                           edges=edge_attr,
                           senders=_s,
                           receivers=_r,
                           n_node=_graph.n_node[idx],
                           n_edge=_graph.n_edge[idx],
                           globals=_graph.globals[idx])

# functions for building padded catalogues
def padded_catalog(catalog, simulator_args=simulator_args):
    pad_nodes_to = simulator_args['pad_nodes_to']

    nodes = jnp.zeros((pad_nodes_to, 7))
    nodes = nodes.at[:catalog.shape[0], :].set(catalog[:pad_nodes_to])

    return nodes, catalog.shape[0]

# build graph "simulator"
def graph_simulator(key, θ, catalog=None,
                    simulator_args=simulator_args,
                    r_connect=0.15,
                    num_halos=None):

    # do mass cut
    if simulator_args["do_noise"]:
        noise = jax.random.normal(key,
                    shape=catalog[:, 0].shape)*simulator_args['noise_scale']*simulator_args["mass_cut"]

        catalog = catalog.at[:, 0].set(catalog[:, 0] + noise)

    # if we want to truncate to a FIXED-length catalog (bad idea !)
    if num_halos is not None:
        masses = catalog[:, 0]
        inds = jnp.argsort(masses)[::-1] # sort largest to smallest
        inds = inds[:num_halos]          # take first few indices
        catalog = catalog[inds]

    else:
        mass_cut = simulator_args["mass_cut"]
        mask = (catalog[:, 0] < mass_cut)
        catalog = jnp.where(mask[:, jnp.newaxis],
                                0.,
                                catalog)

    catalog,n_node = padded_catalog(jnp.squeeze(catalog), simulator_args=simulator_args)

    graph = padded_graph_builder(catalog, θ, simulator_args,\
                                                  r_connect=r_connect)

    return graph

def getgraphs(key, graphs, r_connect=0.15,
              num_halos=None,
              simulator_args=simulator_args, verbose=True):

    num = graphs.nodes.shape[0]
    keys = jax.random.split(key, num=num)
    θs = graphs.globals

    if verbose:
        print('assembling with r_connect = ', r_connect)

    gs = lambda k,θ,cat: graph_simulator(k, θ, cat,
                                         simulator_args=simulator_args,
                                         r_connect=r_connect, num_halos=num_halos)

    graphs = jax.vmap(gs)(keys, θs, graphs.nodes)

    return graphs

# default simulator args
N=512
shape=(N,N,N)
simulator_args = dict(
        N=N,
        L=1.0,
        shape=shape,
        dim=len(shape),
        freqs=None, #get_freqs(shape,1.0),# pre-compute frequencies
        vol_norm=True,           # whether to normalise P(k) by volume
        L_scale=False,            # scale P(k) by length scale
        connect_radius=0.5,
        mass_cut=1.5,             # 10^15 Msun
        do_noise=False,
        nbar=0.0015,    # GIVES US A LOT MORE NODES
        pad_nodes_to=200,
        pad_edges_to= 500, # 1150, 500, 100 for 0.3, 0.2, 0.1
        include_pos=False,
        invert_edges=False,
        squeeze=False,
        get_context=True,
        boxsize=None, # or 1.0001
        return_cat=False,
        node_features=7,
          window=2)
