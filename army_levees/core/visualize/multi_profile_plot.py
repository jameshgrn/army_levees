import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

class MultiProfilePlotter:
    """
    A class to handle plotting multiple elevation profiles in a grid layout.
    """
    def __init__(self, output_dir):
        """
        Initialize the MultiProfilePlotter.
        
        Parameters:
        -----------
        output_dir : str
            Directory where the output plots will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_degradation_profiles(self, filtered_degradation, profiles_per_plot=56):
        """
        Plot elevation profiles for degradation cases in a 7x8 grid
        
        Parameters:
        -----------
        filtered_degradation : pandas.DataFrame
            DataFrame containing the profile data for degradation cases
        profiles_per_plot : int, optional
            Number of profiles to include in each figure (default: 56 for 7x8 grid)
        """
        # Get unique system IDs
        unique_systems = filtered_degradation['system_id'].unique()
        num_systems = len(unique_systems)
        
        # Calculate number of figures needed
        num_figures = math.ceil(num_systems / profiles_per_plot)
        
        for fig_num in range(num_figures):
            # Get the subset of systems for this figure
            start_idx = fig_num * profiles_per_plot
            end_idx = min((fig_num + 1) * profiles_per_plot, num_systems)
            systems_subset = unique_systems[start_idx:end_idx]
            
            # Create and save the figure
            fig = self._create_profile_plot(
                filtered_degradation, 
                'Degradation Elevation Profiles (NLD vs TEP)', 
                fig_num, 
                systems_subset
            )
            
            # Save figure
            filename = f'degradation_profiles_part{fig_num + 1}.png'
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {filename}")
        
    def plot_stable_profiles(self, filtered_stable, profiles_per_plot=56):
        """
        Plot elevation profiles for stable cases in a 7x8 grid
        
        Parameters:
        -----------
        filtered_stable : pandas.DataFrame
            DataFrame containing the profile data for stable cases
        profiles_per_plot : int, optional
            Number of profiles to include in each figure (default: 56 for 7x8 grid)
        """
        # Get unique system IDs
        unique_systems = filtered_stable['system_id'].unique()
        num_systems = len(unique_systems)
        
        # Calculate number of figures needed
        num_figures = math.ceil(num_systems / profiles_per_plot)
        
        for fig_num in range(num_figures):
            # Get the subset of systems for this figure
            start_idx = fig_num * profiles_per_plot
            end_idx = min((fig_num + 1) * profiles_per_plot, num_systems)
            systems_subset = unique_systems[start_idx:end_idx]
            
            # Create and save the figure
            fig = self._create_profile_plot(
                filtered_stable, 
                'Stable Elevation Profiles (NLD vs TEP)', 
                fig_num, 
                systems_subset
            )
            
            # Save figure
            filename = f'stable_profiles_part{fig_num + 1}.png'
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {filename}")

    def plot_aggradation_profiles(self, filtered_aggradation, profiles_per_plot=56):
        """
        Plot elevation profiles for aggradation cases in a 7x8 grid
        
        Parameters:
        -----------
        filtered_aggradation : pandas.DataFrame
            DataFrame containing the profile data for aggradation cases
        profiles_per_plot : int, optional
            Number of profiles to include in each figure (default: 56 for 7x8 grid)
        """
        # Get unique system IDs
        unique_systems = filtered_aggradation['system_id'].unique()
        num_systems = len(unique_systems)
        
        # Calculate number of figures needed
        num_figures = math.ceil(num_systems / profiles_per_plot)
        
        for fig_num in range(num_figures):
            # Get the subset of systems for this figure
            start_idx = fig_num * profiles_per_plot
            end_idx = min((fig_num + 1) * profiles_per_plot, num_systems)
            systems_subset = unique_systems[start_idx:end_idx]
            
            # Create and save the figure
            fig = self._create_profile_plot(
                filtered_aggradation, 
                'Aggradation Elevation Profiles (NLD vs 3DEP)', 
                fig_num, 
                systems_subset
            )
            
            # Save figure
            filename = f'aggradation_profiles_part{fig_num + 1}.png'
            filepath = os.path.join(self.output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"Saved {filename}")

    def _create_profile_plot(self, data, title, fig_num, systems_subset):
        """
        Helper method to create a single figure with multiple profile plots
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the profile data with columns:
            - system_id: USACE system ID
            - elevation: NLD elevation (meters)
            - dep_elevation: 3DEP elevation (meters)
            - distance_along_track: Distance along levee (meters)
        title : str
            Title for the figure
        fig_num : int
            Figure number
        systems_subset : array-like
            Subset of system IDs to plot
        """
        fig, axes = plt.subplots(7, 8, figsize=(32, 28))
        fig.suptitle(title, fontsize=16)
        
        axes_flat = axes.flatten()
        
        for i, system_id in enumerate(systems_subset):
            ax = axes_flat[i]
            
            # Get data for this system
            system_data = data[data['system_id'] == system_id].sort_values('distance_along_track')
            
            # Plot profiles
            # Plot NLD elevation with blue line and dots
            ax.plot(system_data['distance_along_track'], system_data['elevation'], 
                   'b-', linewidth=1, label='NLD')
            ax.plot(system_data['distance_along_track'], system_data['elevation'], 
                   'bo', markersize=3)
            
            # Plot 3DEP elevation with red line and dots
            ax.plot(system_data['distance_along_track'], system_data['dep_elevation'], 
                   'r-', linewidth=1, label='3DEP')
            ax.plot(system_data['distance_along_track'], system_data['dep_elevation'], 
                   'ro', markersize=3)
            
            ax.set_title(f'System ID: {system_id}', fontsize=8)
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=6)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Turn off unused subplots
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for suptitle
        
        return fig

def main():
    parser = argparse.ArgumentParser(description='Plot elevation profiles in grid layout')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed parquet files')
    parser.add_argument('--summary_dir', type=str, default='data/system_id_summary',
                       help='Directory containing classification CSV files')
    parser.add_argument('--output_dir', type=str, default='plots/profiles',
                       help='Directory to save output plots')
    parser.add_argument('--type', type=str, choices=['all', 'degradation', 'aggradation', 'stable'],
                       default='all', help='Type of profiles to plot')
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = MultiProfilePlotter(args.output_dir)
    
    # Process based on type
    if args.type in ['all', 'degradation']:
        degradation_ids = pd.read_csv(os.path.join(args.summary_dir, 'system_id_degradation.csv'))
        degradation_data = []
        for system_id in degradation_ids['system_id']:
            try:
                profile = pd.read_parquet(os.path.join(args.data_dir, f'levee_{system_id}.parquet'))
                degradation_data.append(profile)
            except FileNotFoundError:
                print(f"Warning: Could not find profile data for system {system_id}")
        if degradation_data:
            degradation_data = pd.concat(degradation_data, ignore_index=True)
            plotter.plot_degradation_profiles(degradation_data)
    
    if args.type in ['all', 'stable']:
        stable_ids = pd.read_csv(os.path.join(args.summary_dir, 'system_id_stable.csv'))
        stable_data = []
        for system_id in stable_ids['system_id']:
            try:
                profile = pd.read_parquet(os.path.join(args.data_dir, f'levee_{system_id}.parquet'))
                stable_data.append(profile)
            except FileNotFoundError:
                print(f"Warning: Could not find profile data for system {system_id}")
        if stable_data:
            stable_data = pd.concat(stable_data, ignore_index=True)
            plotter.plot_stable_profiles(stable_data)
    
    if args.type in ['all', 'aggradation']:
        aggradation_ids = pd.read_csv(os.path.join(args.summary_dir, 'system_id_aggradation.csv'))
        aggradation_data = []
        for system_id in aggradation_ids['system_id']:
            try:
                profile = pd.read_parquet(os.path.join(args.data_dir, f'levee_{system_id}.parquet'))
                aggradation_data.append(profile)
            except FileNotFoundError:
                print(f"Warning: Could not find profile data for system {system_id}")
        if aggradation_data:
            aggradation_data = pd.concat(aggradation_data, ignore_index=True)
            plotter.plot_aggradation_profiles(aggradation_data)

if __name__ == "__main__":
    main()