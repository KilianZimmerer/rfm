#!/bin/bash
# This script runs the SUMO simulation for the given configuration file.

# pass the configuration file as an argument
SIMULATION_DIR=$1
if [ -z "$SIMULATION_DIR" ]; then
  echo "No config file provided. Usage: $0 <simulation_dir>"
  exit 1
fi
# Check if the configuration file exists
if [ ! -d "$SIMULATION_DIR" ]; then
  echo "Simulation Dir $SIMULATION_DIR not found!"
  exit 1
fi

sumo -c $SIMULATION_DIR/rail.sumocfg
