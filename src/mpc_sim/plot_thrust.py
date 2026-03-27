"""
MPC Thrust Data Visualization Tool
----------------------------------
This script reads ROS 2 bag files (sqlite3 format), extracts thrust data from 
'/fmu/in/vehicle_rates_setpoint' topic, and generates publication-quality plots.

Features:
- High-resolution (300 DPI) output in PDF, EPS, PNG, and TIFF formats.
- Supports multiple bag files with automatic labeling and comparison.
- Interactive selection of bag files if multiple are found.
- Publication-ready styling (serif fonts, clear labels, grid).

Usage:
    python plot_thrust.py --input path/to/bag_folder --output path/to/save_plots
    
    Arguments:
    --input: Path to a single ROS 2 bag folder or a directory containing multiple bag folders.
    --output: Directory where the generated plots will be saved.
    --formats: List of output formats (e.g., --formats png pdf).

Dependencies:
    pip install -r ../../requirements.txt
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
VEHICLE_RATES_SETPOINT_MSG = """
uint64 timestamp
float32 roll
float32 pitch
float32 yaw
float32[3] thrust_body
float32 reset_integral
"""

VEHICLE_THRUST_SETPOINT_MSG = """
uint64 timestamp
float32[3] xyz
"""

# Register the custom types
add_types = get_types_from_msg(VEHICLE_RATES_SETPOINT_MSG, 'px4_msgs/msg/VehicleRatesSetpoint')
typestore.register(add_types)
add_types = get_types_from_msg(VEHICLE_THRUST_SETPOINT_MSG, 'px4_msgs/msg/VehicleThrustSetpoint')
typestore.register(add_types)

def parse_rosbag(bag_path):
    """
    Parse a single ROS 2 bag file and extract thrust data.
    """
    data = []
    bag_name = Path(bag_path).name
    try:
        with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
            # Possible topics for thrust data (Input for MPC, Output for PID)
            candidate_topics = [
                '/fmu/in/vehicle_rates_setpoint',
                '/fmu/out/vehicle_rates_setpoint',
                '/fmu/out/vehicle_thrust_setpoint'
            ]
            
            # Find which topics actually exist in this bag
            connections = [c for c in reader.connections if c.topic in candidate_topics]
            if not connections:
                print(f"Warning: No thrust-related topics found in {bag_path}")
                return None

            # Sort connections to prioritize certain topics if multiple exist
            # (e.g., prefer RatesSetpoint over ThrustSetpoint if both are there)
            connections.sort(key=lambda c: candidate_topics.index(c.topic))
            best_connection = connections[0]
            print(f"  Reading from topic: {best_connection.topic}")

            for connection, timestamp, rawdata in reader.messages(connections=[best_connection]):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # Extract data based on message type
                if 'VehicleRatesSetpoint' in connection.msgtype:
                    tx, ty, tz = msg.thrust_body
                elif 'VehicleThrustSetpoint' in connection.msgtype:
                    tx, ty, tz = msg.xyz
                else:
                    continue
                
                thrust_magnitude = np.sqrt(tx**2 + ty**2 + tz**2)
                
                data.append({
                    'timestamp': msg.timestamp / 1e6,
                    'thrust_x': tx,
                    'thrust_y': ty,
                    'thrust_z': tz,
                    'thrust_mag': thrust_magnitude,
                    'bag_name': bag_name,
                    'mode': 'PID' if 'out' in connection.topic else 'MPC'
                })
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
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

    # Combine all dataframes
    combined_df = pd.concat(all_data).reset_index(drop=True)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    unique_bags = combined_df['bag_name'].unique()
    
    if len(unique_bags) > 1:
        # Multiple bags: Plot each one as a separate line
        sns.lineplot(data=combined_df, x='time', y='thrust_mag', hue='bag_name', style='mode', ax=ax)
        plt.title('MPC vs PID Thrust Performance Comparison')
    else:
        # Single bag: Plot magnitude and components
        mode = combined_df['mode'].iloc[0]
        sns.lineplot(data=combined_df, x='time', y='thrust_mag', ax=ax, label=f'Total Thrust ({mode})')
        sns.lineplot(data=combined_df, x='time', y='thrust_z', ax=ax, label='Thrust Z-axis', alpha=0.7, linestyle='--')
        plt.title(f'Thrust Performance ({mode}): {unique_bags[0]}')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Thrust [Normalized]')
    ax.legend(loc='best', frameon=True)

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    base_name = 'thrust_analysis'
    if len(unique_bags) == 1:
        base_name += f"_{unique_bags[0]}"
    
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    print(f"\nPlots generated in: {os.path.abspath(output_dir)}")
    plt.show()

def find_bag_folders(base_path):
    """Recursively find folders containing metadata.yaml."""
    bags = []
    p = Path(base_path)
    if (p / 'metadata.yaml').exists():
        bags.append(p)
    else:
        for item in p.iterdir():
            if item.is_dir() and (item / 'metadata.yaml').exists():
                bags.append(item)
    return sorted(bags)

def main():
    parser = argparse.ArgumentParser(description='Process ROS 2 bags and plot thrust data for publication.')
    parser.add_argument('--input', type=str, default=None, help='Path to bag folder or parent directory.')
    parser.add_argument('--output', type=str, default='plots', help='Directory to save plots.')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf', 'eps', 'tiff'], help='Output formats.')
    
    args = parser.parse_args()

    # If no input, use current directory
    search_path = args.input if args.input else '.'
    
    bag_folders = find_bag_folders(search_path)

    if not bag_folders:
        print(f"No ROS 2 bag folders found in '{search_path}'.")
        return

    selected_bags = []
    if len(bag_folders) > 1 and not args.input:
        # Multiple bags found in current dir, let user choose
        print("\nMultiple ROS bags found:")
        for i, folder in enumerate(bag_folders):
            print(f"[{i}] {folder.name}")
        print(f"[{len(bag_folders)}] ALL (Plot all in one graph)")
        
        try:
            choice = input(f"\nSelect a bag to plot (0-{len(bag_folders)}): ").strip()
            idx = int(choice)
            if idx == len(bag_folders):
                selected_bags = bag_folders
            elif 0 <= idx < len(bag_folders):
                selected_bags = [bag_folders[idx]]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Invalid input.")
            return
    else:
        # Single bag found or input path specified
        selected_bags = bag_folders

    print(f"\nProcessing {len(selected_bags)} bag(s)...")

    all_dfs = []
    for bag in selected_bags:
        df = parse_rosbag(str(bag))
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("No valid data extracted.")
        return

    print("Generating plots...")
    plot_results(all_dfs, args.output, args.formats)
    print("Done.")

if __name__ == "__main__":
    main()
