#/bin/bash
# This script generates the SUMO network files for the simulation.
# It uses the netconvert tool to create the network files from the node, edge, and connection files.


# simulation directory as argument
SIMULATION_DIR=$1
# Check if the simulation directory is provided
if [ -z "$1" ]; then
  echo "No simulation directory provided. Usage: $0 <simulation_directory>"
  exit 1
fi
# Check if the simulation directory exists
if [ ! -d "$1" ]; then
  echo "Simulation directory $1 not found!"
  exit 1
fi


netconvert --node-files=$SIMULATION_DIR/rail.nod.xml --edge-files=$SIMULATION_DIR/rail.edg.xml --connection-files=$SIMULATION_DIR/rail.con.xml -o $SIMULATION_DIR/rail.net.xml