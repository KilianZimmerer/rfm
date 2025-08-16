# Railway Foundation Model

A generative dynamic graph model to capture railway systems.


## Setup

```
uv sync
source .venv/bin/activate
```

## The Core Model


## Railway System Simulations

The foundation model is trained on simulated data. The simulations are performed via the simulation tool [sumo](https://sumo.dlr.de/docs/index.html).

Performing simulations is a two step process. We first need to setup the railway network and the run the simulation. 


### 1. Setting up the railway network

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