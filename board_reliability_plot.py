import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from parse_data import parse_data

LIBRARY = 'ZBar'
DATA_FILE = Path(__file__).resolve().parent / 'data' / f'{LIBRARY}_reliability.txt'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

def main():
    data = parse_data(DATA_FILE)
    generate_scatter_plots(data)

def generate_scatter_plots(method_vectors):
    accuracy_means = {}
    accuracy_stds = {}
    methods = []
    
    # Extract the data from the method_vectors dictionary
    for method_name, vectors in method_vectors.items():
        methods.append(method_name)
        accuracy_means[method_name] = vectors['accuracy_means']
        accuracy_stds[method_name] = vectors['accuracy_stds']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'{LIBRARY} Performance for detection approaches')
    
    # Plotting data for the first subset of method_names
    ax1.set_title('Detection Accuracy (All methods)')
    ax1.set_xlabel('Tiles on Board (%)')
    ax1.set_ylabel('Accuracy (%)')
    
    # Plotting data points and error bars for each method in the first subset
    for method_name in methods[:2]:
        accuracy_mean_values = accuracy_means[method_name]
        accuracy_std_values = accuracy_stds[method_name]
        tile_fractions = method_vectors[method_name]['tile_fractions']
        
        # Adjusting error bars to ensure they are between 0 and 100
        upper_error = np.minimum(accuracy_mean_values + 2 * accuracy_std_values, 100) - accuracy_mean_values
        lower_error = accuracy_mean_values - np.maximum(accuracy_mean_values - 2 * accuracy_std_values, 0)
        
        ax1.errorbar(tile_fractions, accuracy_mean_values, yerr=[lower_error, upper_error], fmt='o-', label=method_name)
    
    ax1.legend()
    
    # Plotting data for the second subset of method_names
    ax2.set_title('Detection Accuracy (Iterative only)')
    ax2.set_xlabel('Tiles on Board (%)')
    ax2.set_ylabel('Accuracy (%)')
    
    # Plotting data points and error bars for each method in the second subset
    for method_name in methods[1:]:
        accuracy_mean_values = accuracy_means[method_name]
        accuracy_std_values = accuracy_stds[method_name]
        tile_fractions = method_vectors[method_name]['tile_fractions']
        
        # Adjusting error bars to ensure they are between 0 and 100
        upper_error = np.minimum(accuracy_mean_values + 2 * accuracy_std_values, 100) - accuracy_mean_values
        lower_error = accuracy_mean_values - np.maximum(accuracy_mean_values - 2 * accuracy_std_values, 0)
        
        ax2.errorbar(tile_fractions, accuracy_mean_values, yerr=[lower_error, upper_error], fmt='o-', label=method_name)
    
    ax2.legend()
    
    # Set x-axis limits for both plots
    ax1.set_xlim(-2, 100)
    ax2.set_xlim(-2, 100)
    
    # Set y-axis limits for accuracy plots
    # ax1.set_ylim(97, 100.2)
    # ax2.set_ylim(97, 100.2)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(PLOT_DIR / f'{LIBRARY}_reliability.png')
    plt.close()





if __name__ == '__main__':
    main()
