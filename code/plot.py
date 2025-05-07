import os
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_rewards_from_csv(csv_path):
    rewards = []
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # skip header
        for row in reader:
            rewards.append(float(row[1]))
    return rewards

def load_all_runs_in_subdir(subdir_path):
    runs = []
    for fname in os.listdir(subdir_path):
        if fname.endswith('.csv'):
            run_rewards = load_rewards_from_csv(os.path.join(subdir_path, fname))
            runs.append(run_rewards)
    return runs

def pad_and_stack_runs(runs):
    max_len = max(len(run) for run in runs)
    padded = [np.pad(run, (0, max_len - len(run)), constant_values=np.nan) for run in runs]
    return np.vstack(padded)  # shape: (num_runs, max_len)

def rolling_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def plot_summary_from_directory(parent_dir):
    plt.figure(figsize=(10, 6))

    for subdir_name in sorted(os.listdir(parent_dir)):
        subdir_path = os.path.join(parent_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            continue

        runs = load_all_runs_in_subdir(subdir_path)
        if not runs:
            continue

        stacked = pad_and_stack_runs(runs)  # shape: (runs, episodes)
        mean = np.nanmean(stacked, axis=0)
        min_ = np.nanmin(stacked, axis=0)
        max_ = np.nanmax(stacked, axis=0)

        # Apply rolling average
        mean_smoothed = rolling_average(mean, window=50)
        min_smoothed = rolling_average(min_, window=50)
        max_smoothed = rolling_average(max_, window=50)

        episodes = np.arange(len(mean_smoothed))
        plt.plot(episodes, mean_smoothed, label=subdir_name)
        plt.fill_between(episodes, min_smoothed, max_smoothed, alpha=0.2)

    plt.title("Reward Curves (50-ep rolling avg)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    parent_dir = sys.argv[1]
    plot_summary_from_directory(parent_dir)