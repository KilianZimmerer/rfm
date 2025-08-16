# Railway Foundation Model

The purpose of this project is to make neural networks "*understand*" railway dynamics. In order to achieve this a generative dynamic graph transformer model is used.


## Setup

```
uv sync
source .venv/bin/activate
```

## The Core Model

The core of this project is a **Railway Foundation Model (RFM)** model, built using `PyTorch` and `PyTorch Geometric`. This model is designed to learn the complex dynamics of a railway system by representing the entire network as a heterogeneous graph.

---

### Architecture

The model consists of two main components:

1.  **Graph Representation Learning (RFM)**: The main `RFM` class uses several `HGTConv` layers to process the graph. It takes the features of all nodes (vehicles, track segments, etc.) and the connections between them to compute rich, context-aware embeddings for every element in the system. This allows the model to understand how different components of the railway network influence each other.

2.  **Prediction Head (`Scorer`)**: The `Scorer` is a simple Multi-Layer Perceptron (MLP). After the `RFM` layers generate embeddings, this module takes the embedding of a specific **vehicle** and the embedding of a potential **track segment** to predict the future state.

---

### Prediction Task

For each active vehicle at a given timestep, the model performs a dual prediction task:
* **Track Prediction**: It calculates a score for every track segment in the network, effectively a classification problem to determine which track the vehicle will move to next. This is trained using a **Cross-Entropy Loss**.
* **Position Prediction**: For the predicted track, it calculates the vehicle's relative position on that track segment (a value between 0.0 and 1.0). This is a regression task trained with a **Mean Squared Error (MSE) Loss**.

The total loss is the sum of these two components, allowing the model to be trained end-to-end to predict both *where* a train is going and its *precise location* on that path.

The `main` function orchestrates the entire training and evaluation pipeline, splitting the simulation data, running training epochs, and printing validation metrics for both track accuracy and position error. Once trained, the model weights are saved to a `.safetensors` file.

## Railway System Simulations

The transformer model is trained on simulated data. The simulations are done with [sumo](https://sumo.dlr.de/docs/index.html).

To run simulations you must install the sumo package:

```
sudo apt install sumo sumo-tools sumo-doc
```

Running simulations and storing the outputs is a two step process as described below.


### 1. Setting up the railway network

For this step you will need the [netconvert](https://sumo.dlr.de/docs/netconvert.html) of sumo.

The network setup is done by defining the following files inside a simulations directory `<SIM_DIR>`:

- `rail.nod.xml`: Defines the nodes of the network (i.e. crossings, or signals)
- `rail.edg.xml`: Defines edges (tracks) between these nodes
- `rail.con.xml`: Sets edges as connections

An example can be viewed in [./simulations/sim3](./simulations/sim3/).

The network can then be initialized with:

```bash
bash ./simulations/network <SIM_DIR>
```

Where <SIM_DIR> is the directory where the above files are stored.

The above command creates a file `rail.net.xml` containing all relevant network information from `rail.node.xml`, `rail.edg.xml` and `rail.con.xml`

### 2. Running a simulation

Once the network is defined we further need to define the train routes in `<SIM_DIR>/rail.rou.xml`.

Then the simulation can be started with:

```bash
bash ./simulations/run.sh <SIM_DIR>
```

This will use `<SIM_DIR>/rail.rou.xml` and `<SIM_DIR>/rail.net.xml` and perform the defined dynamcis. The results are stored inside `<SIM_DIR>/output.xml`


## Notes

 - Maybe use an encoder-decoder architecture: The encoder encodes track segments and vehicles. The decoder takes vehicle embedding and current track-embedding as an input and iterates through the track embedding.
 - 