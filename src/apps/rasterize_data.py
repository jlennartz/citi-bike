#!/usr/bin/env python3
"""
Rasterize CitiBike and crash data into 3D grids based on spatial and temporal parameters.

This script processes CitiBike trip data and NYC crash data, creating rasterized 
representations in a 3D grid (x, y, time) for analysis.
"""

import argparse
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*DataFrame.swapaxes.*"
)

# Add src to Python path to import utils
# append the src folder, one level up from this file
src_path = Path(__file__).resolve().parent.parent
sys.path.append(str(src_path))
from utils import (
    setup_grid_parameters,
    process_citibike_data,
    process_crash_data
)


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize CitiBike and crash data into 3D grids"
    )
    parser.add_argument(
        "--cell_x", 
        type=float, 
        default=800.0,
        help="Cell size in x-direction (meters)"
    )
    parser.add_argument(
        "--cell_y", 
        type=float, 
        default=800.0,
        help="Cell size in y-direction (meters)"
    )
    parser.add_argument(
        "--cell_t", 
        type=float, 
        default=6.0,
        help="Cell size in time dimension (hours)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500000,
        help="Batch size for processing CitiBike data"
    )
    
    args = parser.parse_args()
    
    # Hard-coded parameters (as requested)
    lon_min, lat_min, lon_max, lat_max = -74.3, 40.45, -73.6, 40.95
    time_max = 365 * 24  # hours in a year
    ref = pd.Timestamp("2023-01-01 00:00:00")
    crash_data_path = '/home/lennartz/data/nypd-collision/Motor_Vehicle_Collisions_-_Crashes_20250627.csv'
    
    # Create sensible output directory name based on parameters
    dir_name = f"cellx{int(args.cell_x)}m_celly{int(args.cell_y)}m_cellt{int(args.cell_t)}h"
    args.output_dir = f"/home/lennartz/repos/citi-bike/results/{dir_name}"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Setting up grid with parameters:")
    print(f"  Spatial resolution: {args.cell_x}m x {args.cell_y}m")
    print(f"  Temporal resolution: {args.cell_t} hours")
    print(f"  Bounding box: ({lon_min}, {lat_min}) to ({lon_max}, {lat_max})")
    print(f"  Output directory: {args.output_dir}")
    
    # Setup grid parameters
    minx, miny, maxx, maxy, Nx, Ny, Nt, transformer = setup_grid_parameters(
        lon_min, lat_min, lon_max, lat_max, 
        args.cell_x, args.cell_y, args.cell_t, 
        time_max
    )
    
    print(f"Grid dimensions: {Nt} time steps x {Ny} rows x {Nx} columns")
    print(f"Total voxels: {Nt * Ny * Nx:,}")
    
    # Process CitiBike data
    print("\nProcessing CitiBike data...")
    try:
        citibike_counts = process_citibike_data(
            batch_size=args.batch_size,
            ref=ref,
            minx=minx, miny=miny,
            cell_x=args.cell_x, cell_y=args.cell_y, cell_t=args.cell_t,
            transformer=transformer,
            Nx=Nx, Ny=Ny, Nt=Nt
        )
        
        # Save CitiBike data
        citibike_output = os.path.join(args.output_dir, "citibike_counts.npy")
        np.save(citibike_output, citibike_counts)
        print(f"CitiBike data saved to: {citibike_output}")
        print(f"Total CitiBike trips rasterized: {citibike_counts.sum():,}")
        
    except Exception as e:
        print(f"Error processing CitiBike data: {e}")
        return 1
    
    # Process crash data
    print("\nProcessing crash data...")
    try:
        if not os.path.exists(crash_data_path):
            print(f"Warning: Crash data file not found at {crash_data_path}")
            print("Skipping crash data processing...")
        else:
            crash_counts = process_crash_data(
                crash_data_path=crash_data_path,
                ref=ref,
                minx=minx, miny=miny,
                cell_x=args.cell_x, cell_y=args.cell_y, cell_t=args.cell_t,
                transformer=transformer,
                Nx=Nx, Ny=Ny, Nt=Nt
            )
            
            # Save crash data
            crash_output = os.path.join(args.output_dir, "crash_counts.npy")
            np.save(crash_output, crash_counts)
            print(f"Crash data saved to: {crash_output}")
            print(f"Total crashes rasterized: {crash_counts.sum():,}")
            
            # Calculate normalized risk data
            print("\nCalculating normalized risk metrics...")
            
            # Aggregate over time and space
            trips_time_aggregated = citibike_counts.sum(0).astype(np.float32)
            trips_spatial_aggregated = citibike_counts.sum((1,2)).astype(np.float32)
            crashes_time_aggregated = crash_counts.sum(0).astype(np.float32)
            crashes_spatial_aggregated = crash_counts.sum((1,2)).astype(np.float32)
            
            # Normalize (crashes per trip)
            time_normalized = np.nan_to_num(
                crashes_time_aggregated / trips_time_aggregated,
                nan=0.0, posinf=0.0, neginf=0.0
            )
            
            space_normalized = np.nan_to_num(
                crashes_spatial_aggregated / trips_spatial_aggregated,
                nan=0.0, posinf=0.0, neginf=0.0
            )
            
            # Save normalized data
            time_norm_output = os.path.join(args.output_dir, "risk_time_normalized.npy")
            space_norm_output = os.path.join(args.output_dir, "risk_space_normalized.npy")
            
            np.save(time_norm_output, time_normalized)
            np.save(space_norm_output, space_normalized)
            
            print(f"Time-normalized risk saved to: {time_norm_output}")
            print(f"Space-normalized risk saved to: {space_norm_output}")
            
    except Exception as e:
        print(f"Error processing crash data: {e}")
        return 1
    
    # Save metadata
    metadata = {
        'cell_x': args.cell_x,
        'cell_y': args.cell_y, 
        'cell_t': args.cell_t,
        'lon_min': lon_min,
        'lat_min': lat_min,
        'lon_max': lon_max,
        'lat_max': lat_max,
        'minx': minx,
        'miny': miny,
        'maxx': maxx,
        'maxy': maxy,
        'Nx': Nx,
        'Ny': Ny,
        'Nt': Nt,
        'time_max': time_max,
        'reference_time': str(ref),
        'batch_size': args.batch_size
    }
    
    metadata_output = os.path.join(args.output_dir, "metadata.json")
    import json
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to: {metadata_output}")
    print("\nRasterization complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())