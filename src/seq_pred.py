"""Autoregressive evaluation script for the Railway Foundation Model (RFM).

This module evaluates a trained RFM model by performing a sequential rollout
prediction. The main function, `sequential_prediction`, loads a model and an
initial railway system state, then iteratively predicts future vehicle
positions for a specified number of steps.

In each step, the model's output is used to update the graph state, and this
newly generated graph becomes the input for the subsequent prediction. This
allows for an analysis of the model's long-term simulation capabilities.
"""

# TODO: Quantitative Trajectory Evaluation Metrics

from safetensors.torch import load_model
import torch
from src.data import get_data, add_edge
from src.rfm import RFM, get_device
from torch_geometric.data import HeteroData


def sequential_prediction(simulation_id: str, data: HeteroData, prediction_length: int = 10) -> HeteroData:
    """
    Sequentially predicts the next position of vehicles in the simulation.
    Args:
        simulation_id (str): The ID of the simulation.
        data (HeteroData): The input data for the model.
        prediction_length (int): The number of steps to predict.
    Returns:
        HeteroData: The updated data with predicted positions.
    """
    # Load the model
    device = get_device()
    model_test = RFM(
        hidden_channels=60, num_heads=3, num_layers=2,
        node_types=data.node_types, metadata=data.metadata()).to(device)
    # Perform a dummy forward pass to initialize lazy modules
    model_test(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
    load_model(model_test, f"sumo/{simulation_id}/model.safetensors")

    # Sequential prediction
    while prediction_length > 0:
        prediction_length -= 1
        vehicle_outputs = model_test(data.x_dict, data.edge_index_dict, current=data['vehicle'].current)
        del data["vehicle", "precedes", "vehicle"].edge_index

        for idx, vehicle_out in enumerate(vehicle_outputs):
            _, index = vehicle_out["scores"].max(dim=0)
            track_id = vehicle_out["track_ids"][index]
            pos = vehicle_out["pos"][index]
            mask = data["vehicle"].id == idx
            data["vehicle"].x = torch.cat((data["vehicle"].x, pos.view(1, 1)), dim=0)
            data["vehicle"].time = torch.cat((data["vehicle"].time, torch.tensor([[data["vehicle"].time[mask][-1] + 1]])), dim=0)
            data["vehicle"].id = torch.cat((data["vehicle"].id, torch.tensor([[idx]])), dim=0)
            data["vehicle"].current[mask] = False
            data["vehicle"].current = torch.cat((data["vehicle"].current, torch.tensor([[True]])), dim=0)
            data["vehicle", "on", "track"].edge_index = torch.cat((data["vehicle", "on", "track"].edge_index, torch.tensor([[idx], [track_id]])), dim=1)
            data["track", "hosts", "vehicle"].edge_index = torch.cat((data["track", "hosts", "vehicle"].edge_index, torch.tensor([[track_id], [idx]])), dim=1)
            data["vehicle"].predicted = torch.cat((data["vehicle"].predicted, torch.tensor([[True]])), dim=0)

            indices = torch.where(mask)[0]
            for idx in indices[:-2]:
                add_edge(data, "vehicle", "precedes", "vehicle", [idx, indices[-2]])
    return data

if __name__ == "__main__":
    simulation_id = "sim2"
    output_xml = "testoutput.xml"
    batching = {"min": 30, "max": 45}
    data_list, node_mapping = get_data(simulation_id=simulation_id, output_xml=output_xml, batching=batching)
    predicted_data = sequential_prediction(simulation_id=simulation_id, data=data_list[0])
    import pdb;pdb.set_trace()