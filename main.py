import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_scalar_from_event(logdir, tag_prefix='Score', max_steps=5000):
    """Extract scalar values from TensorBoard logs and limit to max_steps"""
    scalar_events = []
    for root, _, files in os.walk(logdir):
        for file in files:
            if "events.out.tfevents" in file:
                for e in tf.compat.v1.train.summary_iterator(os.path.join(root, file)):
                    if len(scalar_events) >= max_steps:
                        break
                    for v in e.summary.value:
                        if v.tag.startswith(tag_prefix):
                            scalar_events.append((e.step, v.simple_value))
                            if len(scalar_events) >= max_steps:
                                break
    print(f"Extracted {len(scalar_events)} events from {logdir}")
    return scalar_events

def average_and_std_runs(runs):
    """Compute the average and standard deviation of scalar events from multiple runs"""
    all_steps = set(step for run in runs for step, _ in run)
    avg_events = []
    std_events = []
    for step in sorted(all_steps):
        values = [value for run in runs for run_step, value in run if run_step == step]
        if values:
            avg_events.append((step, np.mean(values)))
            std_events.append((step, np.std(values) / 2))  # Half the standard deviation for shading
    print(f"Computed average and std for {len(avg_events)} steps")
    return avg_events, std_events

def smooth_data(values, smoothing_factor=0.99):
    """Apply exponential moving average to smooth the data"""
    smoothed_values = []
    last = values[0]
    for value in values:
        smoothed_value = last * smoothing_factor + (1 - smoothing_factor) * value
        smoothed_values.append(smoothed_value)
        last = smoothed_value
    return smoothed_values

# Paths to log directories for agent 1
logdirs_agent1 = [
    '/home/hense/PycharmProjects/Exploration-Enhanced-Contrastive-Learning/logs',
    '/home/hense/PycharmProjects/td3-cl-0529/logs',
    '/home/hense/PycharmProjects/td3-cl-0531/logs'
]

# Paths to log directories for agent 2
logdirs_agent2 = [
    '/home/hense/PycharmProjects/td3-2/logs',
    '/home/hense/PycharmProjects/td3-1/logs',
    '/home/hense/PycharmProjects/Panda-Lift/logs',
]

# Extract scalar events from each run for both agents using the correct tag prefix
runs_agent1 = [extract_scalar_from_event(logdir, tag_prefix='Score - ', max_steps=5000) for logdir in logdirs_agent1]
runs_agent2 = [extract_scalar_from_event(logdir, tag_prefix='Score - ', max_steps=5000) for logdir in logdirs_agent2]

# Compute the average and standard deviation of the runs for both agents
avg_events_agent1, std_events_agent1 = average_and_std_runs(runs_agent1)
avg_events_agent2, std_events_agent2 = average_and_std_runs(runs_agent2)

# Unzip the average events and std events into steps and values for both agents
steps_agent1, values_agent1 = zip(*avg_events_agent1) if avg_events_agent1 else ([], [])
_, std_devs_agent1 = zip(*std_events_agent1) if std_events_agent1 else ([], [])

steps_agent2, values_agent2 = zip(*avg_events_agent2) if avg_events_agent2 else ([], [])
_, std_devs_agent2 = zip(*std_events_agent2) if std_events_agent2 else ([], [])

# Smooth the values for both agents
smoothed_values_agent1 = smooth_data(values_agent1, smoothing_factor=0.995) if values_agent1 else []
smoothed_values_agent2 = smooth_data(values_agent2, smoothing_factor=0.995) if values_agent2 else []

# Plot the smoothed score average and standard deviation for both agents
plt.figure(figsize=(12, 8))

if steps_agent1:
    plt.plot(steps_agent1, smoothed_values_agent1, label='TD3-EECL', color='darkblue')
    plt.fill_between(steps_agent1, np.array(smoothed_values_agent1) - std_devs_agent1, np.array(smoothed_values_agent1) + std_devs_agent1, alpha=0.3, color='lightblue')

if steps_agent2:
    plt.plot(steps_agent2, smoothed_values_agent2, label='TD3', color='darkorange')
    plt.fill_between(steps_agent2, np.array(smoothed_values_agent2) - std_devs_agent2, np.array(smoothed_values_agent2) + std_devs_agent2, alpha=0.3, color='navajowhite')

plt.xlabel('Time Steps')
plt.ylabel('Average Return')
plt.legend()
plt.grid(True)
plt.show()
