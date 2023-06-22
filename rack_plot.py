import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from rack_parse_data import parse_data

LIBRARY = 'multiprocessing'
DATA_FILE = Path(__file__).resolve().parent / 'data' / f'rack_{LIBRARY}.txt'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

def main():
    data = parse_data(DATA_FILE)
    generate_scatter_plots(data)

import matplotlib.pyplot as plt
import numpy as np

def generate_scatter_plots(method_vectors):
    time_means = {}
    time_stds = {}
    accuracy_means = {}
    accuracy_stds = {}
    
    # Extract the data from the method_vectors dictionary
    for method_name, vectors in method_vectors.items():
        time_means[method_name] = vectors['time_means']
        time_stds[method_name] = vectors['time_stds']
        accuracy_means[method_name] = vectors['accuracy_means']
        accuracy_stds[method_name] = vectors['accuracy_stds']
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f'Rack detection performance benchmark results')
    
    # Plotting data for time
    ax1.set_title('Execution Speed')
    ax1.set_xlabel('Tiles on rack')
    ax1.set_ylabel('Time (ms)')
    
    # Plotting data points and error bars for each method
    for method_name, time_mean_values in time_means.items():
        time_std_values = time_stds[method_name]
        tile_fractions = method_vectors[method_name]['tile_fractions']

        # Adjusting error bars to ensure they're not under 0
        upper_error = 2 * time_std_values
        lower_error = time_mean_values - np.maximum(time_mean_values - 2 * time_std_values, 0)

        ax1.errorbar(tile_fractions, time_mean_values, yerr=[lower_error, upper_error], fmt='o-', label=method_name)
    
    ax1.legend(loc="upper left")
    
    # Plotting data for accuracy
    ax2.set_title('Detection Accuracy')
    ax2.set_xlabel('Tiles on Rack')
    ax2.set_ylabel('F1 Score')
    
    # Plotting data points and error bars for each method
    for method_name, accuracy_mean_values in accuracy_means.items():
        accuracy_std_values = accuracy_stds[method_name]
        tile_fractions = method_vectors[method_name]['tile_fractions']
        
        # Adjusting error bars to ensure they are between 0 and 1
        upper_error = np.minimum(accuracy_mean_values + 2 * accuracy_std_values, 1) - accuracy_mean_values
        lower_error = accuracy_mean_values - np.maximum(accuracy_mean_values - 2 * accuracy_std_values, 0)
        
        ax2.errorbar(tile_fractions, accuracy_mean_values, yerr=[lower_error, upper_error], fmt='o-', label=method_name)
    
    ax2.legend()
    
    # Set x-axis limits for both plots
    # ax1.set_xlim(-2, 100)
    # ax2.set_xlim(-2, 100)
    
    # Set y-axis limits for accuracy plot
    ax1.set_ylim(0, 110)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(PLOT_DIR / f'rack_{LIBRARY}_performance.png')
    plt.close()



if __name__ == '__main__':
    main()
