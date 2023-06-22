import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from rack_parse_data import parse_data

LIBRARY = 'ZBar'
DATA_FILE = Path(__file__).resolve().parent / 'data' / f'rack_{LIBRARY}_reliability.txt'
PLOT_DIR = Path(__file__).resolve().parent / 'plots'

def main():
    data = parse_data(DATA_FILE)
    generate_scatter_plots(data)

import matplotlib.pyplot as plt
import numpy as np

def generate_scatter_plots(method_vectors):
    accuracy_means = {}
    accuracy_stds = {}
    
    # Extract the data from the method_vectors dictionary
    for method_name, vectors in method_vectors.items():
        accuracy_means[method_name] = vectors['accuracy_means']
        accuracy_stds[method_name] = vectors['accuracy_stds']
    
    # Create a figure with one subplot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f'Rack detection performance benchmark results')
    
    # Plotting data for accuracy
    ax.set_title('Detection Accuracy')
    ax.set_xlabel('Tiles on Rack')
    ax.set_ylabel('F1 Score')
    
    # Plotting data points and error bars for each method
    for method_name, accuracy_mean_values in accuracy_means.items():
        accuracy_std_values = accuracy_stds[method_name]
        tile_fractions = method_vectors[method_name]['tile_fractions']
        
        # Adjusting error bars to ensure they are between 0 and 1
        upper_error = np.minimum(accuracy_mean_values + 2 * accuracy_std_values, 1) - accuracy_mean_values
        lower_error = accuracy_mean_values - np.maximum(accuracy_mean_values - 2 * accuracy_std_values, 0)
        
        ax.errorbar(tile_fractions, accuracy_mean_values, yerr=[lower_error, upper_error], fmt='o-', label=method_name)
    
    ax.legend()
    
    # Set x-axis limits
    #ax.set_xlim(-2, 100)
    
    # Set y-axis limits for accuracy plot
    #ax.set_ylim(0, 110)
    
    # Adjust spacing
    plt.tight_layout()
    
    # Save the figure to a file
    plt.savefig(PLOT_DIR / f'rack_{LIBRARY}_reliability.png')
    plt.close()




if __name__ == '__main__':
    main()
