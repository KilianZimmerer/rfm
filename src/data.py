"""Parses SUMO simulation data into PyTorch Geometric HeteroData graphs.

This module handles the data ingestion and preprocessing pipeline. It reads raw
XML output files from a SUMO (Simulation of Urban MObility) railway
simulation and transforms them into a list of graph-based data samples
suitable for training a Graph Neural Network.

The process involves two main stages:
1.  Constructing the static railway network graph from a `rail.net.xml` file,
    where railway lanes are represented as 'track' nodes and their connections
    as edges.
2.  Parsing dynamic vehicle movement data from an `output.xml` file and adding
    'vehicle' nodes to the graph for each timestep, linking them to the
    tracks they occupy.

The final output is a list of `HeteroData` objects, where the time-series
vehicle data has been batched into smaller windows. These graphs contain
target labels (`y_track`, `y_pos`) for supervised learning tasks.
"""

import random
import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import HeteroData
from src.utils import BiMap
from collections import defaultdict


def get_data(
        simulation_id: str,
        batching: dict = {"min": 5, "max": 20},
        output_xml: str = "output.xml",
    ) -> tuple[list[HeteroData], BiMap]:
    """Creates a list of HeteroData objects from net_xml and output_xml."""
    net_xml = f"simulations/{simulation_id}/rail.net.xml"
    output_xml = f"simulations/{simulation_id}/{output_xml}"
    data, node_mapping = _data_from_net_xml(net_xml)
    data_list = _add_trains(data, output_xml, node_mapping, batching)
    return data_list, node_mapping

def add_edge(data: HeteroData, src_type: str, rel_type: str, dst_type: str, edge: list):
    """Adds an edge to the HeteroData object."""
    edge_tensor = torch.tensor([edge], dtype=torch.long).T
    data[src_type, rel_type, dst_type].edge_index = (
        torch.cat((data[src_type, rel_type, dst_type].edge_index, edge_tensor), dim=1)
        if data[src_type, rel_type, dst_type]
        else edge_tensor
    )


def _data_from_net_xml(net_xml: str) -> tuple[HeteroData, BiMap]:
    """Creates a HeteroData object from nodes and edges."""
    nodes, edges = _network_from_xml(net_xml)
    data = HeteroData()
    node_mapping = _add_nodes_to_data(data, nodes)
    _add_edges_to_data(data, edges, node_mapping)
    return data, node_mapping


def _add_nodes_to_data(data: HeteroData, nodes: dict) -> BiMap:
    """Adds nodes to the HeteroData object."""
    node_mapping = BiMap()
    for i, node_type in enumerate(nodes):
        for j, lane_id in enumerate(nodes[node_type].keys()):
            value = nodes[node_type][lane_id]
            track_type = "track"
            length_tensor = torch.tensor([[value["length"]]], dtype=torch.float32)
            data[track_type].x = torch.cat((data[track_type].x, length_tensor), dim=0) if data[track_type] else length_tensor
            node_mapping.add(lane_id=lane_id, id_=int(len(data[track_type].x) - 1))
    return node_mapping


def _add_edges_to_data(data: HeteroData, edges: dict, node_mapping: BiMap):
    """Adds edges to the HeteroData object."""
    track_type = "track"
    for start_node, end_nodes in edges.items():
        for end_node in end_nodes:
            edge_tensor = torch.tensor(
                [
                    [node_mapping.get_id(start_node), node_mapping.get_id(end_node)],
                    [node_mapping.get_id(end_node), node_mapping.get_id(start_node)],
                ],
                dtype=torch.long,
            )
            data[track_type, "connects", track_type].edge_index = (
                torch.cat((data[track_type, "connects", track_type].edge_index, edge_tensor), dim=1)
                if data[track_type, "connects", track_type]
                else edge_tensor
            )


def _add_trains(data: HeteroData, output_xml: str, node_mapping: BiMap, batching: dict) -> list[HeteroData]:
    """Adds train nodes to the HeteroData object."""
    output = _parse_fcd_data(output_xml)
    batches = _create_batches(output, batching)
    return [_process_batch(data.clone(), batch, node_mapping) for batch in batches]


def _create_batches(output: list, batching: dict) -> list[list]:
    """Creates batches of data based on the batching configuration, ensuring minimum values per vehicle."""

    # Group output by vehicle_id
    vehicle_data = defaultdict(list)
    for entry in output:
        vehicle_data[entry["vehicle_id"]].append(entry)

    # Create batches ensuring minimum values per vehicle
    batches = []
    while any(vehicle_data.values()):
        batch = []
        for vehicle_id, entries in list(vehicle_data.items()):
            size = random.randint(batching["min"], batching["max"])
            batch.extend(entries[:size])
            vehicle_data[vehicle_id] = entries[size:]
            if not vehicle_data[vehicle_id]:
                del vehicle_data[vehicle_id]
        batches.append(batch)
    return batches


def _process_batch(data: HeteroData, batch: list, node_mapping: BiMap) -> HeteroData:
    """Processes a single batch and updates the HeteroData object."""
    for i, timestep in enumerate(batch):
        _add_vehicle_node(data, timestep, i, node_mapping)
        _add_vehicle_edges(data, timestep, i, node_mapping)
    _update_previous_predictions(data, batch, node_mapping)
    return data


def _add_vehicle_node(data: HeteroData, timestep: dict, index: int, node_mapping: BiMap):
    """Adds a vehicle node to the HeteroData object."""
    track_type = "track"
    node_id = node_mapping.get_id(timestep["lane"])
    rel_pos = timestep["pos"] / data[track_type].x[node_id].item()
    vehicle_data = {
        "x": torch.tensor([[rel_pos]], dtype=torch.float32),
        "time": torch.tensor([[timestep["time"]]], dtype=torch.float32),
        "y_track": torch.tensor([[-1]], dtype=torch.long),
        "y_pos": torch.tensor([[-1]], dtype=torch.float32),
        "id": torch.tensor([[int(timestep["vehicle_id"])]], dtype=torch.long),
        "current": torch.tensor([[False]], dtype=torch.bool),
        "predicted": torch.tensor([[False]], dtype=torch.bool),
    }
    for key, value in vehicle_data.items():
        try:
            data["vehicle"][key] = torch.cat((data["vehicle"][key], value), dim=0)
        except KeyError:
            data["vehicle"][key] = value

def _add_vehicle_edges(data: HeteroData, timestep: dict, index: int, node_mapping: BiMap):
    """Adds edges for the vehicle node."""
    # TODO: there seems to be an error: not all vehicles should connect to the latest but to the last of it's same kind.
    track_type = "track"
    node_id = node_mapping.get_id(timestep["lane"])
    add_edge(data, "vehicle", "on", track_type, [index, node_id])
    add_edge(data, track_type, "hosts", "vehicle", [node_id, index])


def _update_previous_predictions(data: HeteroData, batch: list, node_mapping: BiMap):
    """Updates previous prediction values for the last node."""
    for vehicle_id in data["vehicle"].id.unique():
        mask = data["vehicle"].id == vehicle_id
        indices = torch.where(mask)[0]
        for idx in indices[:-2]:
            add_edge(data, "vehicle", "precedes", "vehicle", [idx, indices[-2]])
        if len(indices) > 1:
            source_index = indices[-1]
            node_id = node_mapping.get_id(batch[source_index]["lane"])
            rel_pos = batch[source_index]["pos"] / data["track"].x[node_id].item()
            target_index = indices[-2]
            data["vehicle"].current[target_index] = torch.tensor([[True]], dtype=torch.bool)
            data["vehicle"].y_track[target_index] = torch.tensor([[node_id]], dtype=torch.long)
            data["vehicle"].y_pos[target_index] = torch.tensor([[rel_pos]], dtype=torch.float32)

def _network_from_xml(net_file: str) -> tuple[dict, dict]:
    """Creates a network representation where lanes are nodes and connections are edges."""
    tree = ET.parse(net_file)
    root = tree.getroot()
    lane_nodes, lane_edges = {}, {}
    for edge in root.findall('.//edge'):
        for lane in edge.findall('.//lane'):
            lane_id = lane.get('id')
            lane_nodes.setdefault(lane_id[0], {})[lane_id] = {'length': float(lane.get('length'))}
    for connection in root.findall('.//connection'):
        start_node = connection.get('from') + '_' + connection.get('fromLane')
        end_node = connection.get('via') or connection.get('to') + '_' + connection.get('toLane')
        lane_edges.setdefault(start_node, []).append(end_node)
    return lane_nodes, lane_edges


def _parse_fcd_data(xml_file: str) -> list[dict]:
    """Parses an fcd-export XML file and returns vehicle data."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return [
        {
            'vehicle_id': vehicle.get('id'),
            'time': float(timestep.get('time')),
            'pos': float(vehicle.get('pos')),
            'lane': vehicle.get('lane'),
            'speed': float(vehicle.get('speed')),
        }
        for timestep in root.findall('timestep')
        for vehicle in timestep.findall('vehicle')
    ]


if __name__ == "__main__":
    simulation_id = "sim1"
    data_list, node_mapping = get_data(simulation_id)
    print(len(data_list))
    print(data_list[0])
