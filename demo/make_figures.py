import os
import pandas as pd
import matplotlib.pyplot as plt


base_dir = os.path.join(os.path.dirname(__file__), 'data')
metrics_data = {}

for subdir in os.listdir(base_dir):
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        method = subdir.split("_")[1]
        trial_dfs = []
        # Look for all trial_{} subdirectories
        for trial_name in os.listdir(subdir_path):
            trial_path = os.path.join(subdir_path, trial_name)
            csv_path = os.path.join(trial_path, 'eval_metrics.csv')
            if os.path.isfile(csv_path):
                df = pd.read_csv(csv_path)
                trial_dfs.append(df[["avg_sum_reward", "pc_success"]])
        metrics_data[method] = trial_dfs

for metric in ["avg_sum_reward", "pc_success"]:
    plt.figure(figsize=(10, 6))

    for method, dfs in metrics_data.items():
        values = [df[metric].values for df in dfs]
        values = pd.DataFrame(values)
        mean_series = values.mean(axis=0)
        std_series = values.std(axis=0)

        steps = [1000 * (i + 1) for i in range(len(mean_series))]

        plt.plot(steps, mean_series, label=f"{method}")
        plt.fill_between(steps, mean_series - std_series, mean_series + std_series, alpha=0.2)

    plt.title(f"{metric} over training steps")
    plt.xlabel("Training Steps")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()

    # Save the figure
    save_path = os.path.join(base_dir, f"{metric}.png")
    plt.savefig(save_path)
    plt.close()