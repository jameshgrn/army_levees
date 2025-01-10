import os
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class MultiProfilePlotter:
    """
    A class to handle plotting multiple elevation profiles in a grid layout.
    """
    def __init__(self, output_dir: str):
        """
        Initialize the MultiProfilePlotter.
        
        Parameters:
        -----------
        output_dir : str
            Directory where the output plots will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profiles_per_page = 56  # 7x8 grid
        
    def plot_degradation_profiles(self, data: pd.DataFrame) -> None:
        """Plot profiles showing degradation, multiple pages if needed."""
        system_ids = data['system_id'].unique()
        num_systems = len(system_ids)
        num_pages = math.ceil(num_systems / self.profiles_per_page)
        
        for page in range(num_pages):
            start_idx = page * self.profiles_per_page
            end_idx = min((page + 1) * self.profiles_per_page, num_systems)
            systems_subset = system_ids[start_idx:end_idx]
            
            title = f"Degradation Profiles (Page {page+1} of {num_pages})"
            fig = self._create_profile_plot(data, title, page, systems_subset)
            fig.savefig(self.output_dir / f"degradation_profiles_page{page+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
    def plot_stable_profiles(self, data: pd.DataFrame) -> None:
        """Plot stable profiles, multiple pages if needed."""
        system_ids = data['system_id'].unique()
        num_systems = len(system_ids)
        num_pages = math.ceil(num_systems / self.profiles_per_page)
        
        for page in range(num_pages):
            start_idx = page * self.profiles_per_page
            end_idx = min((page + 1) * self.profiles_per_page, num_systems)
            systems_subset = system_ids[start_idx:end_idx]
            
            title = f"Stable Profiles (Page {page+1} of {num_pages})"
            fig = self._create_profile_plot(data, title, page, systems_subset)
            fig.savefig(self.output_dir / f"stable_profiles_page{page+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
    def plot_aggradation_profiles(self, data: pd.DataFrame) -> None:
        """Plot profiles showing aggradation, multiple pages if needed."""
        system_ids = data['system_id'].unique()
        num_systems = len(system_ids)
        num_pages = math.ceil(num_systems / self.profiles_per_page)
        
        for page in range(num_pages):
            start_idx = page * self.profiles_per_page
            end_idx = min((page + 1) * self.profiles_per_page, num_systems)
            systems_subset = system_ids[start_idx:end_idx]
            
            title = f"Aggradation Profiles (Page {page+1} of {num_pages})"
            fig = self._create_profile_plot(data, title, page, systems_subset)
            fig.savefig(self.output_dir / f"aggradation_profiles_page{page+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_significant_profiles(self, data: pd.DataFrame) -> None:
        """Plot profiles showing significant changes, multiple pages if needed."""
        system_ids = data['system_id'].unique()
        num_systems = len(system_ids)
        num_pages = math.ceil(num_systems / self.profiles_per_page)
        
        for page in range(num_pages):
            start_idx = page * self.profiles_per_page
            end_idx = min((page + 1) * self.profiles_per_page, num_systems)
            systems_subset = system_ids[start_idx:end_idx]
            
            title = f"Significant Change Profiles (Page {page+1} of {num_pages})"
            fig = self._create_profile_plot(data, title, page, systems_subset)
            fig.savefig(self.output_dir / f"significant_profiles_page{page+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
    def plot_non_significant_profiles(self, data: pd.DataFrame) -> None:
        """Plot profiles showing non-significant changes, multiple pages if needed."""
        system_ids = data['system_id'].unique()
        num_systems = len(system_ids)
        num_pages = math.ceil(num_systems / self.profiles_per_page)
        
        for page in range(num_pages):
            start_idx = page * self.profiles_per_page
            end_idx = min((page + 1) * self.profiles_per_page, num_systems)
            systems_subset = system_ids[start_idx:end_idx]
            
            title = f"Non-significant Change Profiles (Page {page+1} of {num_pages})"
            fig = self._create_profile_plot(data, title, page, systems_subset)
            fig.savefig(self.output_dir / f"non_significant_profiles_page{page+1}.png", dpi=300, bbox_inches='tight')
            plt.close(fig)

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
            ax.plot(system_data['distance_along_track'], system_data['elevation'], 
                   'b-', linewidth=1, label='NLD')
            ax.plot(system_data['distance_along_track'], system_data['elevation'], 
                   'bo', markersize=3)
            
            ax.plot(system_data['distance_along_track'], system_data['dep_elevation'], 
                   'r-', linewidth=1, label='3DEP')
            ax.plot(system_data['distance_along_track'], system_data['dep_elevation'], 
                   'ro', markersize=3)
            
            # Add segment info to title if data is segmented
            if 'segment' in system_data.columns:
                num_segments = system_data['segment'].nunique()
                ax.set_title(f'System ID: {system_id}\n({num_segments} segments)', fontsize=8)
            else:
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

def classify_levees(data: pd.DataFrame, output_dir: str | Path) -> dict[str, list[str]]:
    """
    Classify levees based on elevation differences between NLD and 3DEP.
    
    Classification criteria:
    - Significant: Mean change > 0.1m or < -0.1m
    - Non-significant: Mean change between -0.1m and 0.1m
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    classifications = {
        'significant': [],
        'non_significant': []
    }
    
    # Calculate mean difference for each system
    for system_id in data['system_id'].unique():
        system_data = data[data['system_id'] == system_id]
        mean_diff = (system_data['elevation'] - system_data['dep_elevation']).mean()
        
        if abs(mean_diff) > 0.1:
            classifications['significant'].append(system_id)
        else:
            classifications['non_significant'].append(system_id)
    
    # Save classifications to CSV files
    for category, system_ids in classifications.items():
        if system_ids:  # Only save if we have systems in this category
            df = pd.DataFrame({'system_id': system_ids})
            df.to_csv(output_dir / f'system_id_{category}.csv', index=False)
            logger.info(f"Found {len(system_ids)} {category} profiles")
    
    return classifications

def main():
    parser = argparse.ArgumentParser(description='Plot elevation profiles in grid layout')
    parser.add_argument('--data-dir', type=str, default='data/segments',
                       help='Directory containing filtered segments')
    parser.add_argument('--raw-data', action='store_true',
                       help='Use raw data from data/processed instead of filtered segments')
    parser.add_argument('--type', type=str, choices=['all', 'significant', 'non_significant'],
                       default='all', help='Type of profiles to plot')
    parser.add_argument('--output-dir', type=str, default='plots/profiles',
                       help='Directory to save output plots')
    parser.add_argument('--summary-dir', type=str, default='data/system_id_summary',
                       help='Directory containing classification CSV files')
    
    args = parser.parse_args()
    data_dir = 'data/processed' if args.raw_data else args.data_dir
    
    data = []
    try:
        # Get list of all system IDs
        from army_levees.core.visualize.utils import get_processed_systems
        system_ids = get_processed_systems(data_dir=data_dir)
        
        # Load data for each system
        for system_id in system_ids:
            try:
                if args.raw_data:
                    profile = pd.read_parquet(os.path.join(data_dir, f'levee_{system_id}.parquet'))
                else:
                    segment_files = sorted(Path(data_dir).glob(f'levee_{system_id}_segment_*.parquet'))
                    segments = [pd.read_parquet(f) for f in segment_files]
                    profile = pd.concat(segments, ignore_index=True)
                profile['system_id'] = system_id  # Add system_id column
                data.append(profile)
            except Exception as e:
                logger.warning(f"Could not load data for system {system_id}: {str(e)}")
                continue
        
        # Initialize plotter
        plotter = MultiProfilePlotter(args.output_dir)
        
        if data:
            all_data = pd.concat(data, ignore_index=True)
            
            if args.type == 'all':
                # Create classifications
                logger.info("Classifying levees...")
                classifications = classify_levees(all_data, args.summary_dir)
                
                # Plot each classification type
                for plot_type, system_ids in classifications.items():
                    if not system_ids:
                        logger.warning(f"No {plot_type} profiles found")
                        continue
                        
                    classified_data = all_data[all_data['system_id'].isin(system_ids)]
                    
                    if plot_type == 'significant':
                        plotter.plot_significant_profiles(classified_data)
                    elif plot_type == 'non_significant':
                        plotter.plot_non_significant_profiles(classified_data)
                        
                    logger.info(f"Created plots for {len(system_ids)} {plot_type} profiles")
            else:
                # Create classification if needed
                if not os.path.exists(os.path.join(args.summary_dir, f'system_id_{args.type}.csv')):
                    logger.info("Classifying levees...")
                    classifications = classify_levees(all_data, args.summary_dir)
                    system_ids = classifications[args.type]
                else:
                    # Load existing classification
                    classification_file = os.path.join(args.summary_dir, f'system_id_{args.type}.csv')
                    system_ids = pd.read_csv(classification_file)['system_id'].unique()
                
                if not len(system_ids):
                    logger.error(f"No {args.type} profiles found")
                    return
                    
                classified_data = all_data[all_data['system_id'].isin(system_ids)]
                
                if args.type == 'significant':
                    plotter.plot_significant_profiles(classified_data)
                elif args.type == 'non_significant':
                    plotter.plot_non_significant_profiles(classified_data)
                    
                logger.info(f"Created plots for {len(system_ids)} {args.type} profiles")
        else:
            logger.error("No valid data found to plot")
            
    except Exception as e:
        logger.error(f"Error creating plots: {str(e)}")
        raise

if __name__ == "__main__":
    main()