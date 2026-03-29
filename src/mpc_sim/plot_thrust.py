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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, get_typestore, Stores

# Get the latest typestore
typestore = get_typestore(Stores.LATEST)

# Define PX4 message types for rosbags (Match exactly with .msg files)
VEHICLE_RATES_SETPOINT_MSG = """
uint64 timestamp
float32 roll
float32 pitch
float32 yaw
float32[3] thrust_body
bool reset_integral
"""

VEHICLE_THRUST_SETPOINT_MSG = """
uint64 timestamp
uint64 timestamp_sample
float32[3] xyz
"""

VEHICLE_ATTITUDE_SETPOINT_MSG = """
uint64 timestamp
float32 yaw_sp_move_rate
float32[4] q_d
float32[3] thrust_body
bool reset_integral
bool fw_control_yaw_wheel
"""

# Register the custom types
typestore.register(get_types_from_msg(VEHICLE_RATES_SETPOINT_MSG, 'px4_msgs/msg/VehicleRatesSetpoint'))
typestore.register(get_types_from_msg(VEHICLE_THRUST_SETPOINT_MSG, 'px4_msgs/msg/VehicleThrustSetpoint'))
typestore.register(get_types_from_msg(VEHICLE_ATTITUDE_SETPOINT_MSG, 'px4_msgs/msg/VehicleAttitudeSetpoint'))

def parse_rosbag(bag_path):
    """
    Parse a single ROS 2 bag file and extract thrust data.
    """
    data = []
    bag_path_obj = Path(bag_path)
    bag_name = bag_path_obj.name
    
    try:
        with AnyReader([bag_path_obj], default_typestore=typestore) as reader:
            # Possible topics for thrust data (Input for MPC, Output for PID)
            candidate_topics = [
                '/fmu/in/vehicle_rates_setpoint',
                '/fmu/out/vehicle_rates_setpoint',
                '/fmu/out/vehicle_thrust_setpoint',
                '/fmu/out/vehicle_attitude_setpoint'
            ]
            
            # Find which topics actually exist in this bag
            available_connections = [c for c in reader.connections if c.topic in candidate_topics]
            
            if not available_connections:
                print(f"Warning: No thrust-related topics found in {bag_name}")
                return None

            # Prioritize topics: we prefer the most "final" setpoint available
            # We use the order in candidate_topics as priority
            available_connections.sort(key=lambda c: candidate_topics.index(c.topic))
            best_connection = available_connections[0]
            print(f"  Reading {bag_name} from topic: {best_connection.topic}")

            for connection, timestamp, rawdata in reader.messages(connections=[best_connection]):
                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    
                    # Extract data based on message type
                    if 'VehicleRatesSetpoint' in connection.msgtype:
                        # thrust_body is float32[3]
                        tx, ty, tz = msg.thrust_body
                    elif 'VehicleThrustSetpoint' in connection.msgtype:
                        # xyz is float32[3]
                        tx, ty, tz = msg.xyz
                    elif 'VehicleAttitudeSetpoint' in connection.msgtype:
                        # thrust_body is float32[3]
                        tx, ty, tz = msg.thrust_body
                    else:
                        continue
                    
                    thrust_magnitude = np.sqrt(tx**2 + ty**2 + tz**2)
                    
                    # Determine mode: check folder path or topic
                    mode = 'MPC'
                    full_path_upper = str(bag_path_obj.absolute()).upper()
                    if 'PID' in full_path_upper:
                        mode = 'PID'
                    elif 'OUT' in connection.topic.upper():
                        mode = 'PID'
                    
                    data.append({
                        'timestamp': msg.timestamp / 1e6,
                        'thrust_x': tx,
                        'thrust_y': ty,
                        'thrust_z': tz,
                        'thrust_mag': thrust_magnitude,
                        'bag_name': bag_name,
                        'mode': mode
                    })
                except Exception as e:
                    # Skip corrupted messages within a bag
                    continue
                    
    except Exception as e:
        print(f"Error reading bag {bag_name}: {e}")
        return None

    if not data:
        print(f"No valid messages found in {bag_name}")
        return None
        
    df = pd.DataFrame(data)
    # Normalize timestamp to start from 0 for this bag
    if not df.empty:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df

def plot_results(all_data, output_dir, file_formats=['png', 'pdf', 'eps']):
    """
    Generate publication-quality plots.
    """
    if not all_data:
        print("No data to plot.")
        return

    # Set style for academic journals
    try:
        sns.set_theme(style="whitegrid", font='serif')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        "text.usetex": False,  # Set to True if LaTeX is installed
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": '--'
    })

    # Combine all dataframes
    combined_df = pd.concat(all_data).reset_index(drop=True)
    unique_bags = combined_df['bag_name'].unique()

    # Create figure with two subplots: Magnitude and Components
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Convert to numpy for compatibility
    times = combined_df['time'].to_numpy()
    thrust_mags = combined_df['thrust_mag'].to_numpy()

    # 1. Plot Magnitude
    if len(unique_bags) > 1:
        for bag in unique_bags:
            bag_df = combined_df[combined_df['bag_name'] == bag]
            ax1.plot(bag_df['time'].to_numpy(), bag_df['thrust_mag'].to_numpy(), label=f"{bag} ({bag_df['mode'].iloc[0]})", linewidth=1.5)
        ax1.set_title('Thrust Magnitude Comparison', fontweight='bold')
    else:
        mode = combined_df['mode'].iloc[0]
        ax1.plot(times, thrust_mags, label=f'Total Thrust ({mode})', color='black', linewidth=2)
        ax1.set_title(f'Thrust Performance: {unique_bags[0]}', fontweight='bold')

    ax1.set_ylabel('Thrust Magnitude [Normalized]')
    ax1.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)

    # 2. Plot Components for the first/main bag
    main_bag_df = combined_df[combined_df['bag_name'] == unique_bags[0]]
    m_times = main_bag_df['time'].to_numpy()
    tx = main_bag_df['thrust_x'].to_numpy()
    ty = main_bag_df['thrust_y'].to_numpy()
    tz = main_bag_df['thrust_z'].to_numpy()

    ax2.plot(m_times, tx, label='$T_x$', alpha=0.8, linestyle='-', linewidth=1.2)
    ax2.plot(m_times, ty, label='$T_y$', alpha=0.8, linestyle='-', linewidth=1.2)
    ax2.plot(m_times, tz, label='$T_z$', alpha=0.8, linestyle='--', linewidth=1.5)
    
    ax2.set_title('Thrust Vector Components', fontweight='bold')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Thrust Components')
    ax2.legend(loc='best', frameon=True, fancybox=True, framealpha=0.8)

    # Adjust layout
    plt.tight_layout()

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    base_name = 'thrust_academic_analysis'
    if len(unique_bags) == 1:
        base_name += f"_{unique_bags[0]}"
    
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"{base_name}.{fmt}")
        plt.savefig(save_path, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

    # Energy analysis (simplified)
    # Energy is roughly proportional to sum of squared thrusts over time
    if len(unique_bags) > 0:
        print("\n--- Energy Analysis Summary ---")
        for bag in unique_bags:
            df = combined_df[combined_df['bag_name'] == bag]
            # Simple trapezoidal integration for energy proxy (T^2 * dt)
            dt = df['time'].diff().mean()
            energy_proxy = (df['thrust_mag']**2).sum() * dt
            print(f"Bag: {bag:20} | Mode: {df['mode'].iloc[0]:4} | Energy Proxy: {energy_proxy:.4f}")

    print(f"\nPlots generated in: {os.path.abspath(output_dir)}")
    plt.show()

def find_bag_folders(base_path):
    """Recursively find folders containing metadata.yaml."""
    bags = []
    p = Path(base_path)
    for metadata in p.rglob('metadata.yaml'):
        bags.append(metadata.parent)
    return sorted(list(set(bags)))

def main():
    parser = argparse.ArgumentParser(description='Process ROS 2 bags and plot thrust data for publication.')
    parser.add_argument('--input', type=str, default=None, help='Path to bag folder or parent directory.')
    parser.add_argument('--output', type=str, default='plots', help='Directory to save plots.')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf', 'eps'], help='Output formats.')
    
    args = parser.parse_args()

    # Determine search path: 
    # 1. Use --input if provided
    # 2. Else use current directory '.' (will find all bags in current dir)
    if args.input:
        search_path = args.input
    else:
        search_path = '.'
    
    bag_folders = find_bag_folders(search_path)

    if not bag_folders:
        print(f"No ROS 2 bag folders found in '{search_path}'.")
        return

    selected_bags = []
    # If we found bags in mpc_energy_test or specified path, and it's not a manual interactive session
    if len(bag_folders) > 1 and not args.input:
        # Check if we are in an interactive environment (this is a bit hacky but works for CLI)
        if os.isatty(0):
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
                    print("Invalid selection. Using all bags.")
                    selected_bags = bag_folders
            except (ValueError, EOFError):
                print("Invalid input or non-interactive session. Using all bags.")
                selected_bags = bag_folders
        else:
            # Non-interactive: use all found bags
            print(f"Found {len(bag_folders)} bags. Processing all.")
            selected_bags = bag_folders
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
