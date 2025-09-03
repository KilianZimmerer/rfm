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

#TODO: batching strategy could be improved to time-batching so that there are enoguh vehicles of one type in one batch

import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import HeteroData
from src.utils import BiMap
from collections import defaultdict


def get_data(
        simulation_id: str,
        output_xml: str = "output.xml",
    ) -> tuple[list[HeteroData], BiMap]:
    """Creates a list of HeteroData objects from net_xml and output_xml."""
    net_xml = f"simulations/{simulation_id}/rail.net.xml"
    output_xml = f"simulations/{simulation_id}/{output_xml}"
    data, node_mapping = _data_from_net_xml(net_xml)
    data_list = _add_trains(data, output_xml, node_mapping)
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


def _add_trains(data: HeteroData, output_xml: str, node_mapping: BiMap) -> list[HeteroData]:
    """Adds train nodes to the HeteroData object."""
    output = _parse_fcd_data(output_xml)
    batches = _create_batches(output)
    return [_process_batch(data.clone(), batch, node_mapping) for batch in batches]


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


def _create_batches(
    output: list, 
    window_size: int = 40, 
    step_size: int = 10
) -> list[list]:
    """
    Creates batches using a sliding window over the time-sorted output data.

    Args:
        output (list): The list of all vehicle events, sorted by time.
        window_size (int): The number of events to include in each window (batch).
        step_size (int): The number of events to slide the window forward for the next batch.

    Returns:
        list[list]: A list of batches, where each batch is a list of events
                    from a contiguous time window.
    """
    batches = []
    num_events = len(output)
    
    for i in range(0, num_events - window_size + 1, step_size):
        batch = output[i : i + window_size]
        batches.append(batch)
        
    return batches


def _process_batch(data: HeteroData, batch: list, node_mapping: BiMap) -> HeteroData:
    """Processes a single batch and updates the HeteroData object."""
    for i, timestep in enumerate(batch):
        _add_vehicle_node(data, timestep, i, node_mapping)
        _add_vehicle_edges(data, timestep, i, node_mapping)
    _add_sequential_edges_with_time_diff(data)
    _update_prediction_target(data, batch, node_mapping)
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
    track_type = "track"
    node_id = node_mapping.get_id(timestep["lane"])
    add_edge(data, "vehicle", "on", track_type, [index, node_id])
    add_edge(data, track_type, "hosts", "vehicle", [node_id, index])


def _add_sequential_edges_with_time_diff(data: HeteroData):
    """
    Connects sequential vehicle nodes for each vehicle ID and adds the time
    difference between them as an edge attribute.
    """
    vehicle_nodes = data["vehicle"]
    edge_type = ("vehicle", "precedes", "vehicle")

    # This will store the time difference for each new edge
    time_diffs = []
    
    # We will build the new edge index list here
    new_edge_indices = []

    # Process one vehicle ID at a time
    for vehicle_id in vehicle_nodes.id.unique():
        # Find all nodes belonging to this vehicle
        mask = vehicle_nodes.id == vehicle_id
        indices = torch.where(mask)[0]
        
        # The nodes are already sorted by time because of the batching process.
        # Create edges between consecutive nodes (e.g., n1 -> n2, n2 -> n3, ...)
        for i in range(len(indices) - 1):
            src_node_idx = indices[i]
            dst_node_idx = indices[i+1]
            
            # Add the edge to our list
            new_edge_indices.append([src_node_idx, dst_node_idx])
            # Calculate the time difference and add it to our list
            src_time = vehicle_nodes.time[src_node_idx]
            dst_time = vehicle_nodes.time[dst_node_idx]
            time_diffs.append(dst_time - src_time)

    # If we created any new edges, add them to the data object
    if new_edge_indices:
        data[edge_type].edge_index = torch.tensor(new_edge_indices, dtype=torch.long).T
        data[edge_type].time_diff = torch.tensor(time_diffs, dtype=torch.float32)


def _update_prediction_target(data: HeteroData, batch: list, node_mapping: BiMap):
    """Updates the final node of each vehicle's batch sequence to be the prediction target."""
    for vehicle_id in data["vehicle"].id.unique():
        mask = data["vehicle"].id == vehicle_id
        indices = torch.where(mask)[0]

        if len(indices) > 1:
            last_state_info = batch[indices[-1]]
            node_to_predict_from_idx = indices[-2]

            target_node_id = node_mapping.get_id(last_state_info["lane"])
            target_rel_pos = last_state_info["pos"] / data["track"].x[target_node_id].item()
            
            data["vehicle"].current[node_to_predict_from_idx] = True
            data["vehicle"].y_track[node_to_predict_from_idx] = target_node_id
            data["vehicle"].y_pos[node_to_predict_from_idx] = target_rel_pos


if __name__ == "__main__":
    simulation_id = "sim1"
    data_list, node_mapping = get_data(simulation_id)
    print(len(data_list))
    print(data_list[0])
