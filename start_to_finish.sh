#!/bin/bash
# Capturing data and moving antennas and ends with a reduced beamfitted plot in scape :)

echo "Running raster scan..."
~/scripts/simscan.py
echo "Staging data file..."
/usr/local/bin/augment4.py
echo "Reducing raster scan..."
~/scripts/raster_scan_reduction.py
