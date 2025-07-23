import matplotlib.pyplot as plt
from matplotlib import font_manager

def draw_sign_similarity():
    """
    Draws the sign similarity plot for different datasets and methods.
    """
    # Dataset: [Max Merging, TIES Merging, Max Abs Merging]
    datasets = {
        "CIFAR224": [
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.81, 0.84, 0.87, 0.89, 0.91, 0.91, 0.9, 0.9, 0.88, 0.87],
            [0.78, 0.82, 0.86, 0.88, 0.89, 0.9, 0.9, 0.89, 0.88, 0.87]
        ],
        "ImageNet-R": [
            [0.81, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.81, 0.85, 0.88, 0.9, 0.91, 0.91, 0.9, 0.89, 0.87, 0.86],
            [0.76, 0.81, 0.84, 0.86, 0.87, 0.89, 0.89, 0.89, 0.88, 0.87]
        ],
        "ImageNet-A": [
            [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
            [0.8, 0.84, 0.87, 0.91, 0.93, 0.93, 0.93, 0.91, 0.9, 0.87],
            [0.74, 0.79, 0.83, 0.86, 0.88, 0.88, 0.89, 0.89, 0.88, 0.86]
        ],
        "CUB": [
            [0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81, 0.81],
            [0.77, 0.84, 0.87, 0.9, 0.92, 0.93, 0.92, 0.91, 0.9, 0.89],
            [0.73, 0.81, 0.85, 0.88, 0.89, 0.9, 0.9, 0.9, 0.89, 0.88]
        ],
        "Omnibenchmark": [
            [0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77, 0.77],
            [0.75, 0.8, 0.83, 0.86, 0.89, 0.9, 0.89, 0.87, 0.86, 0.84],
            [0.74, 0.8, 0.83, 0.86, 0.87, 0.88, 0.89, 0.88, 0.88, 0.86]
        ],
        "VTAB": [
            [0.8, 0.79, 0.79, 0.79, 0.79],
            [0.88, 0.9, 0.95, 0.94, 0.9],
            [0.8, 0.82, 0.89, 0.9, 0.88]
        ]
    }

    methods = ['Max Merging', 'TIES Merging', 'Max Absolute Merging']
    colors = ['blue', 'red', 'green']

    # Font settings
    title_font = {'fontsize': 34}
    label_font = {'fontsize': 32}
    tick_fontsize = 30
    legend_font = font_manager.FontProperties(size=30)

    fig, axes = plt.subplots(2, 3, figsize=(32, 16), sharex=False, sharey=False, dpi=1000)
    axes = axes.flatten()

    for i, (dataset_name, values) in enumerate(datasets.items()):
        ax = axes[i]
        for j, method_values in enumerate(values):
            task_indices = list(range(1, len(method_values) + 1))
            ax.plot(task_indices, method_values, marker='o', color=colors[j], label=methods[j])
        
        ax.set_title(dataset_name, **title_font)
        ax.set_xlabel("Task Index", **label_font)
        ax.set_ylabel("Ratio of Sign Similarity", **label_font)
        ax.set_xticks(task_indices)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Add one common legend below all subplots
    fig.legend(methods, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, prop=legend_font)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig("research-ablation_sign_similarity.pdf", dpi=1000, bbox_inches='tight')
    plt.close()


def draw_task_adapting():
    # Data
    methods = ['Max Merging', 'TIES Merging', 'Max Absolute Merging']
    colors = ['blue', 'red', 'green']
    data = {
        "Max Merging": {
            "accuracy": [87.44, 88.58, 88.86, 89.26, 89.23, 89.45, 89.46, 89.77, 89.74, 90.02],
            "forgetting": [4.32, 4.02, 3.98, 3.86, 4.02, 3.99, 3.99, 3.93, 3.98, 3.77],
        },
        "TIES Merging": {
            "accuracy": [87.77, 88.35, 89.01, 89.16, 89.34, 89.42, 89.48, 89.49, 89.56, 89.62],
            "forgetting": [4.4, 4.3, 4.34, 4.5, 4.68, 4.77, 4.91, 5.02, 5.03, 4.98],
        },
        "Max Absolute Merging": {
            "accuracy": [87.77, 88.04, 88.49, 87.83, 87.63, 87.0, 86.41, 85.09, 84.67, 84.64],
            "forgetting": [4.4, 4.62, 5.24, 6.14, 7.04, 8.24, 9.03, 10.74, 11.38, 11.46],
        }
    }
    
    sessions = list(range(1, 11))

    # Font settings
    title_font = {'fontsize': 12}
    label_font = {'fontsize': 12}
    tick_fontsize = 10
    legend_font = font_manager.FontProperties(size=12)

    # Create figure and subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Plot accuracy
    for method, color in zip(methods, colors):
        axes[0].plot(sessions, data[method]['accuracy'], marker='o', color=color, label=method)
    axes[0].set_title("Average Accuracy over Adaptation Sessions", **title_font)
    axes[0].set_ylabel("Average Accuracy", **label_font)
    axes[0].tick_params(axis='both', labelsize=tick_fontsize)
    axes[0].grid(True)
    axes[0].legend(prop=legend_font)

    # Plot forgetting
    for method, color in zip(methods, colors):
        axes[1].plot(sessions, data[method]['forgetting'], marker='x', linestyle='--', color=color, label=method)
    axes[1].set_title("Final Forgetting over Adaptation Sessions", **title_font)
    axes[1].set_xlabel("Adaptation Sessions", **label_font)
    axes[1].set_ylabel("Final Forgetting", **label_font)
    axes[1].tick_params(axis='both', labelsize=tick_fontsize)
    axes[1].grid(True)
    axes[1].legend(prop=legend_font)

    # Final layout adjustments
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig("research-ablation_task_adaptation.pdf", dpi=1600, bbox_inches='tight')
    plt.close()

draw_sign_similarity()
# draw_task_adapting()