import glob
import matplotlib.pyplot as plt
import os
import sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]

clusters = {
    "orion": ['7b_350m', '13b_350m', '7b_7b', '13b_7b', '33b_7b'],
    "phoenix": ['7b_350m', '13b_350m', '7b_7b'],
}
actor_critic = ['7b_350m', '13b_350m', '7b_7b', '13b_7b', '33b_7b', '7b_350m', '13b_350m', '7b_7b']
target = ['dschat', 'base', 'opt']

actor_critic_labels = ['7B/350M', '13B/350M', '7B/7B', '13B/7B', '33B/7B', '7B/350M', '13B/350M', '7B/7B']
target_labels = ['DS-Chat', 'PUZZLE-Base', 'PUZZLE']

data = {t: [] for t in target}
for cluster, acs in clusters.items():
    for ac in acs:
        for t in target:
            # read data from logs with pattern "llama_{ac}_{t}_"
            pattern = f'{input_dir}/{cluster}/llama_{ac}_{t}_'
            files = glob.glob(f'{pattern}*')
            assert len(files) > 0, f'No files found with pattern {pattern}'
            with open(files[0], 'r') as f:
                lines = f.readlines()
                e2e_list = []
                for line in lines:
                    if t in ['base', 'opt']:
                        if 'epoch: ' in line:
                            e2e = float(line.split(' ')[-1]) / 1000
                            e2e_list.append(e2e)
                    else:
                        if "|E2E latency=" in line:
                            # grep the ete latency
                            e2e = line.split("|E2E latency=")[1].split("s")[0]
                            e2e_list.append(float(e2e))
                average_e2e = sum(e2e_list[1:]) / len(e2e_list[1:])
                # print(f'{ac}_{t}: {average_e2e}')
                data[t].append(average_e2e)

# print(data)
# normalize data
_data = {t: data[t].copy() for t in target}
for i, ac in enumerate(actor_critic):
    for t in target:
        _data[t][i] = data['dschat'][i] / _data[t][i]

data = _data
# print(data)

# plot
fig, ax = plt.subplots(figsize=(12, 3.5))
index = range(len(actor_critic))
bar_width = 0.25
opacity = 1
colors = ['#228CC1', '#B4D8E9', '#BEE39C']
for i, t in enumerate(target):
    plt.bar([x + i * bar_width for x in index], data[t], bar_width, alpha=opacity, color=colors[i], label=target_labels[i])
plt.text(1.5, 2.5, f'on orion', fontsize=12, color='black', ha='center')
plt.text(5.5, 2.5, f'on phoenix', fontsize=12, color='black', ha='center')
plt.axvline(x=4.75, color='black', linewidth=1)

plt.ylabel('Speedup')
plt.xticks([x + bar_width for x in index], actor_critic_labels)
plt.ylim(0, 3.5)
plt.legend()
plt.tight_layout()

plt.savefig(f"{output_dir}/figure8.pdf")
