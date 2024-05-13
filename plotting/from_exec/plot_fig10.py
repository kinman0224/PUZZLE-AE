import glob
import matplotlib.pyplot as plt
import sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]

actor_critic = ['7b_7b', '13b_7b']
settings = {
    '7b_7b': {
        'bsz': ['128'],
        'config': ['16_1_1', '8_1_2', '4_1_4', '8_2_1'],
        'config_label': ['(16,1,1)', '(8,1,2)', '(4,1,4)', '(8,2,1)'],
    },
    '13b_7b': {
        'bsz': ['64'],
        'config': ['8_1_2', '4_1_4', '8_2_1'],
        'config_label': ['(8,1,2)', '(4,1,4)', '(8,2,1)'],
    }
}
target = ['wo_bulk', 'w_bulk']

data = {f"{ac}_{b}": [] for ac in actor_critic for b in settings[ac]['bsz']}
data_error = {f"{ac}_{b}": [] for ac in actor_critic for b in settings[ac]['bsz']}
for ac in actor_critic:
    for b in settings[ac]['bsz']:
        tag = f"{ac}_{b}"
        for c in settings[ac]['config']:
            d = {}
            d_full = {t: [] for t in target}
            for t in target:
                pattern = f'{input_dir}/llama_{ac}_{b}_{c}_{t}'
                files = glob.glob(f'{pattern}*')
                assert len(files) > 0, f'No files found with pattern {pattern}'
                with open(files[0], 'r') as f:
                    lines = f.readlines()
                    e2e_list = []
                    for line in lines:
                        if 'epoch: ' in line:
                            e2e = float(line.split(' ')[-5]) / 1000
                            e2e_list.append(e2e)
                    d_full[t] = e2e_list[1:]
                    average_e2e = sum(e2e_list[1:]) / len(e2e_list[1:])
                    # print(f'7b_7b_{b}_{c}_{t}: {average_e2e}')
                    d[t] = average_e2e
            average_speedup = d_full['wo_bulk'][-1] / d_full['w_bulk'][-1]
            speedup_list = [wo / w for wo, w in zip(d_full['wo_bulk'], d_full['w_bulk'])]
            error_list = [abs(average_speedup - spu) for spu in speedup_list]
            # print(f'llama_{ac}_{b}_{c}: {[f"{e:.3f}" for e in error_list]}')
            # print(f'llama_{ac}_{b}_{c}_{t}: {[f"{spu:.3f}" for spu in speedup_list]}')
            # print(f'llama_{ac}_{b}_{c}: speedup {average_speedup:.3f}')
            data[tag].append(d['wo_bulk'] / d['w_bulk'])
            data_error[tag].append(max(error_list))

# print(data)
theoretical = {
    '7b_7b': {'128': [1.2548, 1.1483, 1.0586, 1.2682]},
    '13b_7b': {'64': [1.1955, 1.1012, 1.346531]}
}

# plot (2,2) subplots, each corresponds to a batch size
fig, axs = plt.subplots(2, 1, figsize=(8, 6))
bar_width = 0.3
opacity = 1
colors = ['#228CC1', '#BEE39C']

for row, ac in enumerate(actor_critic):
    bsz = settings[ac]['bsz']
    index = [idx - 1 for idx in (range(len(settings[ac]['config'])))]
    config_label = settings[ac]['config_label']
    for b in bsz:
        tag = f"{ac}_{b}"
        axs[row].bar(index, data[tag], bar_width, alpha=opacity, color=colors[1], label='PUZZLE', yerr=data_error[tag], capsize=5)
        axs[row].bar(index, [max(th-d, 0) for th, d in zip(theoretical[ac][b], data[tag])], bar_width, bottom=data[tag], alpha=opacity, color=colors[0], label='Theoretical', capsize=5)
        axs[row].text(0.9, 1.2, f'batch size {b}', fontsize=12, color='black', ha='center')
        axs[row].set_ylabel('Speedup')
        axs[row].set_xticks([x + bar_width for x in index])
        axs[row].set_xticklabels(config_label)
        axs[row].set_ylim(0, 1.5)
        axs[row].grid(axis='y', linestyle='-.')
        # axs[0].set_title(f'Batch size {b}')

bars = []
labels = ['Theoretical', 'PUZZLE']
for ax in axs.flat:
    bars.append(ax.bar([0, 1], [0, 0], color=colors, alpha=opacity))
fig.legend(bars[0], labels, bbox_to_anchor=(0.5, 0.96), loc='upper center', ncol=2, fontsize=12)
plt.savefig(f'{output_dir}/figure10.pdf')
