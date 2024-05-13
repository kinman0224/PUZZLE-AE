import glob
import matplotlib.pyplot as plt
import os
import sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]

clusters = ['orion', 'phoenix']
actor_critic = ['7b_350m', '7b_7b', '13b_7b']
target = ['wo_bulk', 'w_bulk']

actor_critic_labels = ['7B/350M', '7B/7B', '13B/7B']
target_labels = ['PUZZLE w/o ts', 'PUZZLE w/ ts']

data = {t: [] for t in target}
for c in clusters:
    for ac in actor_critic:
        for t in target:
            # read data from logs with pattern "llama_{ac}_{t}_"
            pattern = f'{input_dir}/{c}/llama_{ac}_{t}_'
            files = glob.glob(f'{pattern}*')
            assert len(files) > 0, f'No files found with pattern {pattern}'
            with open(files[0], 'r') as f:
                lines = f.readlines()
                e2e_list = []
                for line in lines:
                    if 'epoch: ' in line:
                        e2e = float(line.split(' ')[-5]) / 1000
                        e2e_list.append(e2e)
                average_e2e = sum(e2e_list[1:]) / len(e2e_list[1:])
                # print(f'{ac}_{t}: {average_e2e}')
                data[t].append(average_e2e)

# print(data)
# normalize data
_data = {t: data[t].copy() for t in target}
for i, ac in enumerate(actor_critic * len(clusters)):
    for t in target:
        _data[t][i] = data['wo_bulk'][i] / _data[t][i]

data = _data
# print(data)

# plot
fig, ax = plt.subplots(figsize=(8, 3.5))
index = range(len(actor_critic) * len(clusters))
bar_width = 0.3
opacity = 1
# colors = ['#228CC1', '#B4D8E9', '#BEE39C']
colors = ['#228CC1', '#BEE39C']
for i, t in enumerate(target):
    plt.bar([x + i * bar_width for x in index], data[t], bar_width, alpha=opacity, color=colors[i], label=target_labels[i])
plt.text(5., 1.2, 'on phoenix', fontsize=12, color='black', ha='center')
plt.text(0.5, 1.2, 'on orion', fontsize=12, color='black', ha='center')
plt.axvline(x=2.75, color='black', linewidth=1)
plt.ylabel('Speedup')
plt.xticks([x + bar_width for x in index], actor_critic_labels * len(clusters))
plt.ylim(0, 1.3)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2)
plt.grid(axis='y', linestyle='-.')
plt.tight_layout()

plt.savefig(f"{output_dir}/figure9.pdf")
