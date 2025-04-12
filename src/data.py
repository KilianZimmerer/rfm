import random
import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from utils import BiMap


def get_data(simulation_id: str, batching: dict = {"min": 5, "max": 20}) -> list[HeteroData]:
    """Creates a list of HeteroData objects from net_xml and output_xml.
    
    Args:
        net_xml (str): Path to the SUMO network XML file.
        output_xml (str): Path to the SUMO simulation fcd-export xml file that contains simulation output.
        batching (dict): Dictionary containing min and max values for batching.

    Returns:
        List of HeteroData obejcts
    """

    net_xml = f"sumo/{simulation_id}/rail.net.xml"
    output_xml = f"sumo/{simulation_id}/output.xml"
    data, node_mapping = _data_from_net_xml(net_xml=net_xml)
    data_list = _add_trains(data, output_xml=output_xml, node_mapping=node_mapping, batching=batching)
    return data_list, node_mapping

def _data_from_net_xml(net_xml: str) -> HeteroData:
    """Creates a HeteroData object from nodes and edges
    
    Args:
        net_xml (str): Path to the SUMO network XML file.
        

    Returns:
        HeteroData obejct
    """

    nodes, edges = _network_from_xml(net_xml)

    data = HeteroData()

    node_mapping = BiMap()
    for i, node_type in enumerate(nodes):
        for j, lane_id in enumerate(nodes[node_type].keys()):
            value = nodes[node_type][lane_id]
            track_type = "track"
            if not data[track_type]:
                data[track_type].x = torch.tensor([[value["length"]]], dtype=torch.float32)            
            else:
                data[track_type].x = torch.cat((data[track_type].x, torch.tensor([[value["length"]]], dtype=torch.float32)), dim=0)
            node_mapping.add(lane_id=lane_id, id_=int(i * len(nodes[node_type]) + j))

    for start_node, end_nodes in edges.items():
        track_type = "track"
        for end_node in end_nodes:
            if not data[track_type, "connects", track_type]:
                data[track_type, "connects", track_type].edge_index = torch.tensor(
                    [
                        [node_mapping.get_id(start_node), node_mapping.get_id(end_node)],
                        [node_mapping.get_id(end_node), node_mapping.get_id(start_node)],
                    ],
                    dtype=torch.long
                )
            else:
                data[track_type, "connects", track_type].edge_index = torch.cat(
                    (
                        data[track_type, "connects", track_type].edge_index,
                        torch.tensor(
                            [
                                [node_mapping.get_id(start_node), node_mapping.get_id(end_node)],
                                [node_mapping.get_id(end_node), node_mapping.get_id(start_node)],
                            ],
                            dtype=torch.long)
                        ),
                        dim=1
                    )
    return data, node_mapping


def _add_trains(data: HeteroData, output_xml: str, node_mapping: dict, batching: dict = {"min": 5, "max": 20}) -> list[HeteroData]:
    """Add train nodes to the HeterData Object.
    
    Args:
        data:
        output_xml (str): Path to the SUMO simulation fcd-export xml file that contains simulation output.

    Returns:
        HeteroData object containing train nodes from simulation output.
    """

    # read output_xml file and parse 
    output = _parse_fcd_data(output_xml)
    batches = []
    while True:
        size = random.randint(batching["min"], batching["max"])
        batches.append(output[:size])
        output = output[size:]
        if len(output) < batching["min"]:
            break
        if not output:
            break
    
    data_list = []
    for batch in batches:
        clone = data.clone()
        for i, timestep in enumerate(batch):
            track_type = "track"
            node_id = node_mapping.get_id(timestep["lane"])
            rel_pos = timestep["pos"] / clone[track_type].x[node_id].item()
            if not clone["vehicle"]:
                clone["vehicle"].x = torch.tensor([[rel_pos]], dtype=torch.float32)            
                clone["vehicle"].time = torch.tensor([[timestep["time"]]], dtype=torch.float32)
                clone["vehicle"].y_track = torch.tensor([[-1]], dtype=torch.long)
                clone["vehicle"].y_pos = torch.tensor([[-1]], dtype=torch.float32)
                clone["vehicle"].id = torch.tensor([[int(timestep["vehicle_id"])]], dtype=torch.long)
                clone["vehicle"].current = torch.tensor([[False]], dtype=torch.bool)
            else:
                clone["vehicle"].x = torch.cat((clone["vehicle"].x, torch.tensor([[rel_pos]], dtype=torch.float32)), dim=0)
                clone["vehicle"].time = torch.cat((clone["vehicle"].time, torch.tensor([[timestep["time"]]], dtype=torch.float32)), dim=0)
                clone["vehicle"].y_track = torch.cat((clone["vehicle"].y_track, torch.tensor([[-1]], dtype=torch.long)), dim=0)
                clone["vehicle"].y_pos = torch.cat((clone["vehicle"].y_pos, torch.tensor([[-1]], dtype=torch.float32)), dim=0)
                clone["vehicle"].id = torch.cat((clone["vehicle"].id, torch.tensor([[int(timestep["vehicle_id"])]], dtype=torch.float32)), dim=0)
                clone["vehicle"].current = torch.cat((clone["vehicle"].current, torch.tensor([[False]], dtype=torch.bool)), dim=0)

            if not clone["vehicle", "on", track_type]:
                clone["vehicle", "on", track_type].edge_index = torch.tensor(
                    [
                        [i],
                        [node_id],
                    ],
                   dtype=torch.long,
                )
            else:    
                clone["vehicle", "on", track_type].edge_index = torch.cat(
                    (
                        clone["vehicle", "on", track_type].edge_index,
                        torch.tensor(
                            [
                                [i],
                                [node_id],
                            ],
                            dtype=torch.long
                        )
                    ),
                    dim=1
                )
            if not clone[track_type, "hosts", "vehicle"]:
                clone[track_type, "hosts", "vehicle"].edge_index = torch.tensor(
                    [
                        [node_id],
                        [i],
                    ],
                   dtype=torch.long,
                )
            else:    
                clone[track_type, "hosts", "vehicle"].edge_index = torch.cat(
                    (
                        clone[track_type, "hosts", "vehicle"].edge_index,
                        torch.tensor(
                            [
                                [node_id],
                                [i],
                            ],
                            dtype=torch.long
                        )
                    ),
                    dim=1
                )
            if not clone["vehicle", "precedes", "vehicle"]:
                clone["vehicle", "precedes", "vehicle"].edge_index = torch.tensor(
                    [
                        [i],
                        [len(batch) - 1],
                    ],
                   dtype=torch.long,
                )
            elif i != len(batch) - 1:    
                clone["vehicle", "precedes", "vehicle"].edge_index = torch.cat(
                    (
                        clone["vehicle", "precedes", "vehicle"].edge_index,
                        torch.tensor(
                            [
                                [i],
                                [len(batch) - 1],
                            ],
                            dtype=torch.long
                        )
                    ),
                    dim=1
                )
        # fill previous prediction value to the last node as true value
        for id in clone["vehicle"].id.unique():
            mask = clone["vehicle"].id == id
            indices = torch.where(mask)[0]
            if len(indices) > 1:
                source_index = indices[-1]
                node_id = node_mapping.get_id(batch[source_index]["lane"])
                rel_pos = batch[source_index]["pos"] / clone["track"].x[node_id].item()
                target_index = indices[-2]
                clone["vehicle"].current[target_index] = torch.tensor([[True]], dtype=torch.bool)
                clone["vehicle"].y_track[target_index] = torch.tensor([[node_id]], dtype=torch.long)
                clone["vehicle"].y_pos[target_index] = torch.tensor([[rel_pos]], dtype=torch.float32)
        data_list.append(clone)
    return data_list


def _network_from_xml(net_file):
    """Creates a network representation where lanes are nodes and connections are edges.

    Args:
        net_file (str): Path to the SUMO network XML file.

    Returns:
        tuple: A tuple containing two dictionaries:
            - lane_nodes: A dictionary where keys are lane IDs and values are dictionaries 
                         containing 'max_speed', 'length', and 'type'.
            - lane_edges: A list with tuples representing lane connections 
                          (from_lane, to_lane).
    """

    tree = ET.parse(net_file)
    root = tree.getroot()

    lane_nodes = {}
    lane_edges = {}

    for edge in root.findall('.//edge'):
        for lane in edge.findall('.//lane'):
            lane_id = lane.get('id')
            if not lane_id[0] in lane_nodes:
                lane_nodes[lane_id[0]] = {}

            lane_nodes[lane_id[0]][lane_id] = {
                'length': float(lane.get('length')),
            }

    for connection in root.findall('.//connection'):
        start_node = connection.get('from') + '_' + connection.get('fromLane')
        if connection.get('via'):
            end_node = connection.get('via')
        else:
            end_node = connection.get('to') + '_' + connection.get('toLane')
        try:
            lane_edges[start_node].append(end_node)
        except KeyError:
            lane_edges[start_node] = [end_node]
    return lane_nodes, lane_edges


def _parse_fcd_data(xml_file: str):
    """
    Parses an fcd-export XML file and returns vehicle data in a list of dictionaries.

    Args:
        xml_file (str): Path to the fcd-export XML file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a vehicle's data at a specific timestep.  Returns an empty list if no timesteps are found.  
              Each dictionary contains 'vehicle_id', 'time', 'pos', 'lane', and 'speed'.
    """

    tree = ET.parse(xml_file)
    root = tree.getroot()
    vehicle_data = []
    for timestep in root.findall('timestep'):
        time = float(timestep.get('time'))
        for vehicle in timestep.findall('vehicle'):
            vehicle_data.append({
                'vehicle_id': vehicle.get('id'),
                'time': time,
                'pos': float(vehicle.get('pos')),
                'lane': vehicle.get('lane'),
                'speed': float(vehicle.get('speed'))
            })
    return vehicle_data


if __name__ == "__main__":
    simulation_id = "sim1"
    data_list = get_data(simulation_id)
    print(len(data_list))
    print(data_list[0])
