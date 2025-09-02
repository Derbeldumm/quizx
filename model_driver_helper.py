# model_driver_helper.py

import torch
import torch.nn.functional as F
from enum import Enum
import igraph as ig

from torch_geometric.data import Data
from typing import Tuple, List
import numpy as np


class Decomp(Enum):
    """Enumeration of decomposition types."""

    CAT = 0
    MAGIC5 = 1
    CUT = 2
    PAIR_CUT = 3
    MAGIC5INIT = 4


vertices_per_decomp = {
    Decomp.CAT: 1,
    Decomp.CUT: 1,
    Decomp.MAGIC5: 5,
    Decomp.MAGIC5INIT: 1,
    Decomp.PAIR_CUT: 2,
}


def quizx_to_pyg(zx_diagram, random_feature_dim: int = 0) -> Data:
    """Convert a graph-like ZX diagram to a PyG Data instance."""
    vertices = zx_diagram.vertices()

    # Given we are working with Clifford + T graphs, store
    # phases as their multiples of pi/4 (after modulo 2*pi)
    vertex_phases = zx_diagram.phases()
    phases = torch.tensor([int(vertex_phases[v] * 4) % 8 for v in vertices])
    # print(f"Phases: {phases}")
    # Use one hot encoding for the phases.
    # These are our node features.
    x = F.one_hot(phases.long(), num_classes=8).float()

    decomp_mask = {}
    decomp_mask[Decomp.CAT.name] = torch.tensor(
        [
            vertex_phases[v] % 1 == 0
            and len(zx_diagram.neighbors(v)) in (3, 4, 5, 6)
            and all(vertex_phases[n] % 0.5 == 0.25 for n in zx_diagram.neighbors(v))
            for v in vertices
        ]
    ).bool()
    decomp_mask[Decomp.MAGIC5.name] = torch.tensor(
        [vertex_phases[v] % 0.5 == 0.25 for v in vertices]
    ).bool()
    decomp_mask[Decomp.PAIR_CUT.name] = torch.tensor(
        [vertex_phases[v] % 0.5 == 0.25 for v in vertices]
    ).bool()
    decomp_mask[Decomp.CUT.name] = torch.ones(len(vertices)).bool()

    # Create tensor with edge connectivity information.
    # edges = [
    #     (source, target)
    #     for source in vertices
    #     for target in zx_diagram.neighbors(source)
    # ]
    edges = zx_diagram.edges()
    # print(edges)

    edge_index = torch.tensor(
        [
            [source for source, _ in edges] + [target for _, target in edges],
            [target for _, target in edges] + [source for source, _ in edges],
        ],
        dtype=torch.int64,
    )

    obs = Data(x=x, edge_index=edge_index, decomp_mask=decomp_mask)
    add_random_node_features(obs, random_feature_dim)
    add_struct_features(obs, len(vertices), edges)
    # add_individual_node_features(obs)
    return obs


def add_random_node_features(graph: Data, random_feature_dim):
    random_feature = torch.rand(
        graph.x.shape[0], random_feature_dim, device=graph.x.device
    )
    # augment samples with Random Node Initialization
    graph.x = torch.cat((graph.x, random_feature), dim=1)


# def add_individual_node_features(graph: Data):
#     vertex_feature = torch.range(1,graph.x.shape[0], device=graph.x.device).unsqueeze(1)
#     size_feature = torch.ones(graph.x.shape[0], device=graph.x.device).unsqueeze(1)*graph.x.shape[0]
#     # augment samples with Random Node Initialization
#     graph.x = torch.cat((graph.x, vertex_feature, size_feature), dim=1)


def add_struct_features(graph: Data, n, edges):
    g = ig.Graph(n=n, edges=set(edges), directed=False)
    biconnected_components, articulation_points = g.biconnected_components(
        return_articulation_points=True
    )
    membership_vector = biconnected_components.membership
    component_sizes = biconnected_components.sizes()

    articulation_feature = torch.tensor(
        [1 if v.index in articulation_points else 0 for v in g.vs]
    )

    biconnection_features = [torch.zeros(n) for i in range(3)]

    for i in range(n):
        node_bcc_indices = membership_vector[i]
        node_bcc_sizes = [component_sizes[idx] for idx in node_bcc_indices]

        # print(node_bcc_indices, i)

        biconnection_features[0][i] = len(node_bcc_indices)
        biconnection_features[1][i] = np.mean(node_bcc_sizes)
        biconnection_features[2][i] = np.std(node_bcc_sizes)

    amount_cycles = 6
    subcounts = [torch.zeros(n) for i in range(amount_cycles)]
    for size in range(amount_cycles):
        cycle = ig.Graph.Ring(size + 3, circular=True)
        induced_mappings = g.get_subisomorphisms_lad(cycle, induced=True)
        subgraph_node_sets = {frozenset(mapping) for mapping in induced_mappings}
        for subgraph_node_set in subgraph_node_sets:
            for vertex in subgraph_node_set:
                subcounts[size][vertex] += 1

    graph.structure = torch.stack(
        [articulation_feature] + biconnection_features + subcounts, dim=1
    )

    # fig, ax = plt.subplots()
    # ig.plot(
    #         g,
    #         target=ax,
    #         vertex_size=30,
    #         vertex_color="lightblue",
    #         vertex_label=range(n),
    #         vertex_frame_color = ["red" if v in articulation_points else "black" for v in g.vs],
    #         edge_width=0.8,
    #         edge_color='gray'
    # )
    # plt.savefig("dummy_name.png")


def load_model(model_path: str):
    """Loads the pickled PyTorch model."""
    print(f"Loading model from: {model_path}")  
    # The model was saved with torch.save(model, ...), so we load it directly.
    # Ensure the SupervisedModel class definition is available in the environment.
    try:
        model = torch.load(model_path, weights_only=False)
    except Exception as err:
        print("Error when loading model", err)
    print("Loading suceess!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model


def run_model_on_graph(model, rust_graph) -> Tuple[str, List[int]]:
    """
    Converts a quizx graph from Rust, runs the model, and returns the chosen decomposition.
    """
    if len(rust_graph.vertices())<5:
        return ("CUT", [0])
    # 1. Convert the graph from quizx format to pyg format
    pyg_data = quizx_to_pyg(rust_graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pyg_data.to(device)

    # 3. Run the model's forward pass
    # The model returns a list of tuples: [(Decomp, [vertex], score, ...)]
    # We are interested in the first result for a single graph.
    with torch.no_grad():
        result = model(pyg_data)

    # 4. Extract the decomposition type and vertices
    # Example result: [(Decomp.CUT, [15], 0, 0, 0)]
    decomp_obj, vertices, _, _, _ = result[0]

    # print(decomp_obj)
    # print(vertices)

    # We need to return simple types (str, list) back to Rust
    decomp_name = decomp_obj.name  # e.g., "CUT"

    return (decomp_name, vertices)
    # return ("CUT", [0])


def run_estimator_model_on_graph(model, rust_graph, selection) -> Tuple[str, List[int]]:
    """
    Converts a quizx graph from Rust, runs the model, and returns the chosen decomposition.
    """
    # 1. Convert the graph from quizx format to pyg format
    pyg_data = quizx_to_pyg(rust_graph)

    #2. Add the selection feature
    pyg_data.selection = torch.zeros(pyg_data.x.shape[0])
    for vertex in selection:
        pyg_data.selection[vertex] = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pyg_data.to(device)

    # 3. Run the model's ranking pass
    with torch.no_grad():
        result = model.rank(pyg_data)

    return result.item()
    # return ("CUT", [0])