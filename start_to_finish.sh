#!/bin/bash
# The famous start to finish script.
# Starts with capturing data and moving antennas
# and ends with a reduced beamfitted plot in scape :)

echo "Running raster scan..."
~/scripts/data_raster_scan.py
echo "Staging data file..."
/usr/local/bin/augment4.py
echo "Reducing raster scan..."
~/scripts/data_raster_reduction.py
