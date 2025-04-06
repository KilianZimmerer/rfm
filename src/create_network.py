import xml.etree.ElementTree as ET
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


def data_from_net_xml(net_xml: str) -> HeteroData:
    """Creates a HeteroData object from nodes and edges
    
    Args:
        net_xml (str): Path to the SUMO network XML file.
        

    Returns:
        HeteroData obejct
    """

    nodes, edges = _network_from_xml(net_xml)

    data = HeteroData()

    node_mapping = {}
    for type in nodes:
        for id, lane_id in enumerate(nodes[type].keys()):
            value = nodes[type][lane_id]
            if not data[type]:
                data[type].x = torch.tensor([[value["length"]]], dtype=torch.float32)            
            else:
                data[type].x = torch.cat((data[type].x, torch.tensor([[value["length"]]], dtype=torch.float32)), dim=0)
            node_mapping[lane_id] = {"id": id, "type": type}

    for start_node, end_nodes in edges.items():
        start_type = node_mapping[start_node]["type"]
        for end_node in end_nodes:
            end_type = node_mapping[end_node]["type"]
            if not data[start_type, "connects", end_type]:
                data[start_type, "connects", end_type].edge_index = torch.tensor(
                    [[node_mapping[start_node]["id"]], [node_mapping[end_node]["id"]]], dtype=torch.long
                )
            else:
                data[start_type, "connects", end_type].edge_index = torch.cat(
                    (
                        data[start_type, "connects", end_type].edge_index,
                        torch.tensor([[node_mapping[start_node]["id"]], [node_mapping[end_node]["id"]]], dtype=torch.long)), dim=1
                    )
    # Give the freedom to gather information of both directions.
    data = T.ToUndirected()(data)
    return data


def add_trains(data: HeteroData, output_xml: str) -> HeteroData:
    """Add train nodes to the HeterData Object.
    
    Args:
        data:
        output_xml (str): Path to the SUMO simulation output file:

    Returns:
        HeteroData object containing train nodes
    """
    return data

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


if __name__ == "__main__":
    net_file = "sumo/sim1/rail.net.xml"
    data = data_from_net_xml(net_file)
    data = add_trains(data)
    print(data)
    import pdb;pdb.set_trace()