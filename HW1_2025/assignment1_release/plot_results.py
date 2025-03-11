import os
import json
import argparse
import matplotlib.pyplot as plt

def generate_plots(list_of_dirs, legend_names, save_path, title_suffix=""):
    """ 
    Generate accuracy/loss plots according to log files.

    :param list_of_dirs: List of paths to log directories.
    :param legend_names: List of legend names corresponding to each result directory.
    :param save_path: Path to save the generated figures.
    :param title_suffix: Optional suffix for the plot title.
    """
    assert len(list_of_dirs) == len(legend_names), "Names and log directories must have the same length"

    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        assert os.path.exists(json_path), f"No json file in {logdir}"
        with open(json_path, 'r') as f:
            data[name] = json.load(f)
    
    os.makedirs(save_path, exist_ok=True)  # Ensure save directory exists
    
    # Define metrics to plot
    for yaxis in ['train_accs', 'valid_accs', 'train_losses', 'valid_losses']:
        plt.figure(figsize=(12, 10))  # 调整图像大小
        for name in data:
            plt.plot(data[name][yaxis], label=name)

        plt.xlabel('Epochs')
        plt.ylabel(yaxis.replace('_', ' ').title())  
        plt.title(f"{yaxis.replace('_', ' ').title()} vs Epochs {title_suffix}")
        plt.grid()

        # 放置 legend 到图表下方
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8, frameon=False)

        plt.tight_layout()  # 调整布局避免 legend 与图重叠
        plt.savefig(os.path.join(save_path, f'{yaxis}.png'), bbox_inches='tight')
        plt.show()

def plot_training_time(list_of_dirs, legend_names, save_path, title_suffix=""):
    """
    Generate training time comparison plot.

    :param list_of_dirs: List of paths to log directories.
    :param legend_names: List of legend names corresponding to each result directory.
    :param save_path: Path to save the generated figure.
    :param title_suffix: Optional suffix for the plot title.
    """
    data = {}
    for logdir, name in zip(list_of_dirs, legend_names):
        json_path = os.path.join(logdir, 'results.json')
        if not os.path.exists(json_path):
            print(f"⚠️ Warning: No JSON file found in {logdir}, skipping...")
            continue
        with open(json_path, 'r') as f:
            results = json.load(f)
        if "train_times" in results:
            data[name] = sum(results["train_times"])  # Total training time

    # Plot training time as a bar chart
    plt.figure(figsize=(8, 6))
    plt.bar(data.keys(), data.values(), color=plt.cm.Paired.colors[:len(data)])
    plt.xlabel('Configuration')
    plt.ylabel('Total Training Time (seconds)')
    plt.title(f"Total Training Time Comparison {title_suffix}")
    plt.grid(axis='y')
    plt.xticks(rotation=30, ha="right")
    plt.savefig(os.path.join(save_path, 'training_time.png'))
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plot results from multiple training runs.")
    parser.add_argument(
        "--result_dirs", nargs='+', required=True, 
        help="List of result directories containing results.json files."
    )
    parser.add_argument(
        "--legend_names", nargs='+', required=True, 
        help="List of legend names corresponding to each result directory."
    )
    parser.add_argument(
        "--save_path", type=str, default="./plots", 
        help="Directory to save the generated plots (default: ./plots)"
    )
    parser.add_argument(
        "--title", type=str, default="", 
        help="Optional title suffix for the plots (default: '')"
    )
    parser.add_argument(
        "--only_time", action="store_true", 
        help="If set, only plot training time instead of all metrics."
    )
    
    args = parser.parse_args()

    # Generate plots based on the --only_time flag
    if args.only_time:
        plot_training_time(args.result_dirs, args.legend_names, args.save_path, title_suffix=args.title)
    else:
        generate_plots(args.result_dirs, args.legend_names, args.save_path, title_suffix=args.title)
        plot_training_time(args.result_dirs, args.legend_names, args.save_path, title_suffix=args.title)
