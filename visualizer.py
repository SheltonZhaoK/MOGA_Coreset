import os, cv2, re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['svg.fonttype'] = 'none'
colors3 = [
    "#D6594C", "#4D7787", "#AAB083", "#E88C1F", 
    "#7D8995", "#E9BD27", "#7262ac", "#046586", "#28A9A1", "#C9A77C", "#F4A016",'#F6BBC6','#E71F19',
]

# def plot_coreset(train_data, train_labels, core_indices, method_name, data_name):
#     df = pd.DataFrame(train_data[:,[0,1]], columns=['Feature 1', 'Feature 2'])
#     df['Label'] = train_labels

#     # Start the plot
#     plt.figure(figsize=(10, 8))

#     # Plot the original training distribution with hue based on the label
#     sns.scatterplot(data=df, x='Feature 1', y='Feature 2', hue='Label', facecolors='none', edgecolor="muted")

#     # Overlay the coreset points in a uniform color
#     # Assuming 'core_indices' is a list of indices for the coreset points
#     plt.scatter(train_data[core_indices, 0], train_data[core_indices, 1], color='red', label='Coreset', marker='x', s=5)

#     # Customize the plot
#     plt.title(f'Coreset Selection by {method_name}')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend(title='Label / Coreset')
#     plt.savefig(os.path.join("/deac/csc/khuriGrp/zhaok220/thesis/output/training_dist", f'coreset_{method_name}_reference_{data_name}.png'))
#     plt.close()

def plot_coreset(full_data, full_labels, coresets_dict, configs, args, data):
    if not isinstance(full_data, pd.DataFrame):
        full_data = pd.DataFrame(full_data, columns=[f'Feature {i+1}' for i in range(full_data.shape[1])])
    full_data['Label'] = full_labels

    methods_order = [
        "Full", "EvoCore", "UniformSampling", "Kmeans",  # First row
        "MOGA_naive", "MOGA_distance", "MOGA_balance", "MOGA_preserve",  # Second row
        "MOGA_d+b", "MOGA_distance_nf1", "MOGA_balance_nf1", "MOGA_preserve_nf1",  # Third row
    ]

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, method_name in enumerate(methods_order):
        ax = axes[idx]
        # Plot the full dataset in the background
        sns.scatterplot(data=full_data, x='Feature 1', y='Feature 2', hue='Label', palette=colors3, alpha=0.1, legend=False, ax=ax)
        # Overlay the coreset points
        if method_name in coresets_dict:
            coreset_indices = coresets_dict[method_name]
            coreset_data = full_data.iloc[coreset_indices]
            sns.scatterplot(data=coreset_data, x='Feature 1', y='Feature 2', hue='Label', palette=colors3, edgecolor='k', s=80, legend=(idx == 0), ax=ax)

        ax.set_title(method_name)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        if idx == 0:  # Only show legend for the first plot
            ax.legend(title='Label', loc='upper right')

    plt.tight_layout()
    plt.savefig(f"/deac/csc/khuriGrp/zhaok220/thesis/output/scatterPlots/{args.i}/{data}_{configs['seed']}.png")
    plt.savefig(f"/deac/csc/khuriGrp/zhaok220/thesis/output/scatterPlots/{args.i}/{data}_{configs['seed']}.svg")

def plot_fitness_evolution(fitness_list, name):
    num_generations = len(fitness_list)
    num_objectives = len(fitness_list[0])

    # Create a figure with subplots for each fitness objective
    fig, axs = plt.subplots(num_objectives, 1, figsize=(10, num_objectives * 5), squeeze=False)
    axs = axs.flatten()  # Flatten in case there's only one subplot

    # Set a title for the whole figure
    fig.suptitle('Evolution of Fitness Objectives Over Generations', fontsize=16)

    # Iterate over each fitness objective
    for i in range(num_objectives):
        max_values = [generation[i][0] for generation in fitness_list]  # Extract max values for this objective
        min_values = [generation[i][1] for generation in fitness_list]  # Extract min values for this objective
        avg_values = [generation[i][2] for generation in fitness_list]  # Extract avg values for this objective

        generations = range(1, num_generations + 1)
        axs[i].plot(generations, max_values, label='Max', marker='o', linestyle='-', color='r')
        axs[i].plot(generations, min_values, label='Min', marker='x', linestyle='--', color='b')
        axs[i].plot(generations, avg_values, label='Avg', marker='s', linestyle='-.', color='g')

        axs[i].set_title(f'Objective {i+1}')
        axs[i].set_xlabel('Generation')
        axs[i].set_ylabel('Fitness Value')
        axs[i].legend()
        axs[i].grid(True)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"/deac/csc/khuriGrp/zhaok220/thesis/output/fitnesses/fitness_{name}_reference.png")

#python3 ensemble_training.py -d reference -s 100
def plot_fitness_scatter(fitness_list, gen, name):
    # Determine the number of fitness dimensions
    num_dimensions = len(fitness_list[0])
    
    # Create a DataFrame from the fitness list
    if num_dimensions == 2:
        df = pd.DataFrame(fitness_list, columns=['Fitness 1', 'Fitness 2'])
        sns.scatterplot(data=df, x='Fitness 1', y='Fitness 2')
        plt.title(f"Generation {gen}")
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.ylim((0, 70))
        plt.xlim((0.2, 0.90))
    elif num_dimensions == 3:
        df = pd.DataFrame(fitness_list, columns=['Fitness 1', 'Fitness 2', 'Fitness 3'])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['Fitness 1'], df['Fitness 2'], df['Fitness 3'])
        ax.set_xlabel('Fitness 1')
        ax.set_ylabel('Fitness 2')
        ax.set_zlabel('Fitness 3')
        plt.title('3D Scatter Plot of Optimal Solutions Fitness')
    else:
        print("Unsupported number of fitness dimensions")
        return
    
    # Save the plot to a file
    plt.savefig(f"/deac/csc/khuriGrp/zhaok220/thesis/output/fitnesses_scatter/{name}_fitness_reference_{num_dimensions}d_gen{gen}.svg")
    plt.close()

def create_video():
    image_folder = '/deac/csc/khuriGrp/zhaok220/thesis/output/fitnesses_scatter'
    output_video = 'fitness_evolution.mp4'

    # Function to extract the generation number from the filename
    def sort_key(filename):
        match = re.search(r'gen(\d+)', filename)
        return int(match.group(1)) if match else 0

    # List and sort image files based on the generation number
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images_sorted = sorted(images, key=sort_key)

    # Ensure there are images to process
    if not images_sorted:
        print("No images found in the directory.")
        return

    # Read the first frame to establish video properties
    frame = cv2.imread(os.path.join(image_folder, images_sorted[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, 5, (width, height))

    # Write each frame to the video
    for image in images_sorted:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    video.release()

# Call the function to create the video
create_video()

