"""
MPC vs PID Current & Energy Consumption Analysis Tool
---------------------------------------------------
This script reads ROS 2 bag files, extracts battery current data from 
'/fmu/out/battery_status' topic, and generates publication-quality plots.

Features:
- Compares current consumption between different algorithms (MPC vs PID).
- Calculates total energy consumption in Ampere-hours (Ah).
- High-resolution (300 DPI) output in PDF, EPS, and PNG formats.
- Recursive bag searching and automatic mode detection.

Usage:
    python3 plot_current.py --input . --output current_plots
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rosbags.highlevel import AnyReader
from rosbags.typesys import get_types_from_msg, get_typestore, Stores

# Get the latest typestore
typestore = get_typestore(Stores.LATEST)

# Define PX4 BatteryStatus message type (Full definition to match serialized data)
BATTERY_STATUS_MSG = """
uint64 timestamp
bool connected
float32 voltage_v
float32 current_a
float32 current_average_a
float32 discharged_mah
float32 remaining
float32 scale
float32 time_remaining_s
float32 temperature
uint8 cell_count
uint8 source
uint8 priority
uint16 capacity
uint16 cycle_count
uint16 average_time_to_empty
uint16 serial_number
uint16 manufacture_date
uint16 state_of_health
uint16 max_error
uint8 id
uint16 interface_error
float32[14] voltage_cell_v
float32 max_cell_voltage_delta
bool is_powering_off
bool is_required
uint16 faults
uint8 warning
float32 full_charge_capacity_wh
float32 remaining_capacity_wh
uint16 over_discharge_count
float32 nominal_voltage
float32 internal_resistance_estimate
float32 ocv_estimate
float32 ocv_estimate_filtered
float32 volt_based_soc_estimate
float32 voltage_prediction
float32 prediction_error
float32 estimation_covariance_norm
"""

# Register the custom type
add_types = get_types_from_msg(BATTERY_STATUS_MSG, 'px4_msgs/msg/BatteryStatus')
typestore.register(add_types)

def parse_rosbag(bag_path):
    """
    Parse a single ROS 2 bag file and extract battery current data.
    """
    data = []
    bag_name = Path(bag_path).name
    try:
        with AnyReader([Path(bag_path)], default_typestore=typestore) as reader:
            topic = '/fmu/out/battery_status'
            connections = [c for c in reader.connections if c.topic == topic]
            
            if not connections:
                print(f"Warning: Topic {topic} not found in {bag_path}")
                return None

            # Determine mode based on folder name or parent folder name
            mode = 'MPC' # Default
            full_path_upper = str(Path(bag_path).absolute()).upper()
            
            if 'PID' in full_path_upper:
                mode = 'PID'
            elif 'MPC' in full_path_upper:
                mode = 'MPC'
            elif 'OUT' in connections[0].topic:
                mode = 'PID' # PX4 internal controller topics usually contain 'out'

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                
                # PX4 current is usually positive for discharge
                current = msg.current_a
                if current < 0: current = 0 # Filter out unknown/invalid values
                
                data.append({
                    'timestamp': msg.timestamp / 1e6,
                    'current': current,
                    'voltage': msg.voltage_v,
                    'power': current * msg.voltage_v,
                    'bag_name': bag_name,
                    'mode': mode
                })
    except Exception as e:
        print(f"Error reading {bag_path}: {e}")
        return None

    if not data:
        return None
        
    df = pd.DataFrame(data)
    if not df.empty:
        df['time'] = df['timestamp'] - df['timestamp'].iloc[0]
    return df

def plot_results(all_data, output_dir, file_formats=['png', 'pdf', 'eps']):
    """
    Generate publication-quality current, voltage, and power plots.
    """
    if not all_data:
        print("No data to plot.")
        return

    # Set style for academic journals
    sns.set_theme(style="whitegrid", font='serif')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": '--'
    })

    combined_df = pd.concat(all_data).reset_index(drop=True)
    unique_bags = combined_df['bag_name'].unique()

    # Create figure with three subplots: Current, Voltage, and Power
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Use a consistent color palette for algorithms
    palette = sns.color_palette("bright", len(unique_bags))

    # 1. Plot Current
    for i, bag in enumerate(unique_bags):
        df = combined_df[combined_df['bag_name'] == bag]
        label = f"{df['mode'].iloc[0]} ({bag})"
        # Convert to numpy to avoid pandas indexing errors
        ax1.plot(df['time'].to_numpy(), df['current'].to_numpy(), label=label, linewidth=1.5, color=palette[i])
    
    ax1.set_title('Battery Current Consumption Comparison', fontweight='bold')
    ax1.set_ylabel('Current [A]')
    ax1.legend(loc='best', frameon=True)

    # 2. Plot Voltage
    for i, bag in enumerate(unique_bags):
        df = combined_df[combined_df['bag_name'] == bag]
        ax2.plot(df['time'].to_numpy(), df['voltage'].to_numpy(), linewidth=1.5, color=palette[i])
    
    ax2.set_title('Battery Voltage Drop Analysis', fontweight='bold')
    ax2.set_ylabel('Voltage [V]')

    # 3. Plot Power
    for i, bag in enumerate(unique_bags):
        df = combined_df[combined_df['bag_name'] == bag]
        ax3.plot(df['time'].to_numpy(), df['power'].to_numpy(), linewidth=1.5, color=palette[i])
    
    ax3.set_title('Instantaneous Power Consumption ($P = V \times I$)', fontweight='bold')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Power [W]')

    plt.tight_layout()

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    for fmt in file_formats:
        save_path = os.path.join(output_dir, f"energy_analysis_comparison.{fmt}")
        plt.savefig(save_path, format=fmt, bbox_inches='tight')
        print(f"Saved: {save_path}")

    # --- Quantitative Energy Analysis ---
    print("\n" + "="*60)
    print(f"{'Algorithm Mode':<15} | {'Bag Name':<20} | {'Energy (Wh)':<12}")
    print("-" * 60)
    
    results = []
    for bag in unique_bags:
        df = combined_df[combined_df['bag_name'] == bag]
        # Calculate Energy (Wh) = Sum(Power * dt) / 3600
        # Use simple rectangular integration
        dt = df['time'].diff().mean()
        energy_wh = (df['power'].sum() * dt) / 3600.0
        mode = df['mode'].iloc[0]
        print(f"{mode:<15} | {bag:<20} | {energy_wh:.4f}")
        results.append({'mode': mode, 'energy': energy_wh})
    
    # If we have both MPC and PID, calculate percentage improvement
    if any(r['mode'] == 'MPC' for r in results) and any(r['mode'] == 'PID' for r in results):
        mpc_energy = next(r['energy'] for r in results if r['mode'] == 'MPC')
        pid_energy = next(r['energy'] for r in results if r['mode'] == 'PID')
        if pid_energy > 0:
            improvement = (pid_energy - mpc_energy) / pid_energy * 100
            print("-" * 60)
            print(f"MPC Energy Saving vs PID: {improvement:.2f}%")
        else:
            print("-" * 60)
            print("MPC Energy Saving vs PID: N/A (PID energy is 0)")
        
    print("="*60)

    print(f"\nAcademic analysis plots generated in: {os.path.abspath(output_dir)}")
    plt.show()

def find_bag_folders(base_path):
    """Recursively find folders containing metadata.yaml."""
    bags = []
    p = Path(base_path)
    # Use rglob to find all metadata.yaml files in subdirectories
    for metadata in p.rglob('metadata.yaml'):
        bags.append(metadata.parent)
    return sorted(list(set(bags))) # Use set to avoid duplicates and sort for consistency

def main():
    parser = argparse.ArgumentParser(description='Analyze battery current from ROS 2 bags.')
    parser.add_argument('--input', type=str, default='.', help='Path to bag(s).')
    parser.add_argument('--output', type=str, default='energy_plots', help='Output dir.')
    args = parser.parse_args()

    bag_folders = find_bag_folders(args.input)
    if not bag_folders:
        print("No bags found.")
        return

    all_dfs = []
    for bag in bag_folders:
        print(f"Reading: {bag.name}...")
        df = parse_rosbag(str(bag))
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        plot_results(all_dfs, args.output)

if __name__ == "__main__":
    main()