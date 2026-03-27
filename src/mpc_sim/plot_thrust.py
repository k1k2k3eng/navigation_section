"""
MPC Thrust Data Visualization Tool
----------------------------------
This script reads ROS 2 bag files (sqlite3 format), extracts thrust data from 
'/fmu/in/vehicle_rates_setpoint' topic, and generates publication-quality plots.

Features:
- High-resolution (300 DPI) output in PDF, EPS, PNG, and TIFF formats.
- Supports multiple bag files with automatic mean and standard deviation shading.
- Publication-ready styling (serif fonts, clear labels, grid).

Usage:
    python plot_thrust.py --input path/to/bag_folder --output path/to/save_plots
    
    Arguments:
    --input: Path to a single ROS 2 bag folder or a directory containing multiple bag folders.
    --output: Directory where the generated plots will be saved.
    --formats: List of output formats (e.g., --formats png pdf).

Dependencies:
    pip install pandas numpy matplotlib seaborn scipy rosbags pyyaml
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, get_typestore, Stores

# Get the latest typestore
typestore = get_typestore(Stores.LATEST)

# Define PX4 message types for rosbags
# Based on PX4 message definitions
VEHICLE_RATES_SETPOINT_MSG = """
uint64 timestamp
float32 roll
float32 pitch
float32 yaw
float32[3] thrust_body
float32 reset_integral
"""

# Register the custom types
add_types = get_types_from_msg(VEHICLE_RATES_SETPOINT_MSG, 'px4_msgs/msg/VehicleRatesSetpoint')
typestore.register(add_types)

def parse_rosbag(bag_path):
    """
    Parse a single ROS 2 bag file and extract thrust data.
    """
    data = []
    try:
        # AnyReader works with a list of bag paths
        with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
            topic_name = '/fmu/in/vehicle_rates_setpoint'
            
            # Find the connection for the topic
            connections = [c for c in reader.connections if c.topic == topic_name]
            if not connections:
                print(f"Warning: Topic {topic_name} not found in {bag_path}")
                return None

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # Extract data
                # thrust_body is float32[3] -> [x, y, z]
                # Usually for multirotors, thrust is mainly in z
                thrust_x = msg.thrust_body[0]
                thrust_y = msg.thrust_body[1]
                thrust_z = msg.thrust_body[2]
                thrust_magnitude = np.sqrt(thrust_x**2 + thrust_y**2 + thrust_z**2)
                
                data.append({
                    'timestamp': msg.timestamp / 1e6,  # Convert to seconds if it's microseconds
                    'thrust_x': thrust_x,
                    'thrust_y': thrust_y,
                    'thrust_z': thrust_z,
                    'thrust_mag': thrust_magnitude,
                    'roll_rate': msg.roll,
                    'pitch_rate': msg.pitch,
                    'yaw_rate': msg.yaw
                })
    except Exception as e:
        import traceback
        print(f"Error reading {bag_path}: {e}")
        traceback.print_exc()
        return None

    if not data:
        return None
        
    df = pd.DataFrame(data)
    # Normalize timestamp to start from 0
    if not df.empty:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df

def plot_results(all_data, output_dir, file_formats=['png', 'pdf', 'eps', 'tiff']):
    """
    Generate publication-quality plots.
    """
    if not all_data:
        print("No data to plot.")
        return

    # Set style for academic journals
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
        "savefig.dpi": 300
    })

    # Combine all dataframes into one for seaborn processing
    combined_df = pd.concat(all_data, keys=range(len(all_data)), names=['run_id', 'index']).reset_index()

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot thrust magnitude with error shades if multiple runs exist
    if len(all_data) > 1:
        sns.lineplot(data=combined_df, x='time', y='thrust_mag', ax=ax, label='Thrust Magnitude (Mean)', errorbar='sd')
        plt.title('MPC Thrust Performance (Multiple Runs)')
    else:
        sns.lineplot(data=combined_df, x='time', y='thrust_mag', ax=ax, label='Thrust Magnitude')
        # Also plot components if single run
        sns.lineplot(data=combined_df, x='time', y='thrust_z', ax=ax, label='Thrust Z', alpha=0.7, linestyle='--')
        plt.title('MPC Thrust Performance')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Thrust [Normalized]')
    ax.legend(loc='best', frameon=True)

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    base_name = 'thrust_analysis'
    
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Process ROS 2 bags and plot thrust data for publication.')
    parser.add_argument('--input', type=str, default='mpc_energy_test', help='Path to the directory containing ROS 2 bag folders.')
    parser.add_argument('--output', type=str, default='plots', help='Directory to save the generated plots.')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf', 'eps', 'tiff'], help='Output formats (png, pdf, eps, tiff).')
    
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path {input_path} does not exist.")
        return

    # Find all ROS 2 bags (folders containing metadata.yaml)
    bag_folders = []
    if (input_path / 'metadata.yaml').exists():
        bag_folders.append(input_path)
    else:
        # Search subdirectories
        for item in input_path.iterdir():
            if item.is_dir() and (item / 'metadata.yaml').exists():
                bag_folders.append(item)

    if not bag_folders:
        print(f"No ROS 2 bag folders found in {input_path}")
        return

    print(f"Found {len(bag_folders)} bag(s). Parsing...")

    all_dfs = []
    for bag in bag_folders:
        df = parse_rosbag(str(bag))
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("Failed to extract data from any bag.")
        return

    print("Generating plots...")
    plot_results(all_dfs, args.output, args.formats)
    print("Done.")

if __name__ == "__main__":
    main()
