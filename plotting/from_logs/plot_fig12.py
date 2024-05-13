import glob
import matplotlib.pyplot as plt
import re
import numpy as np
import sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]

clusters = ['orion', 'phoenix']
target = ['dschat', 'base', 'opt']
target_labels = ['DSChat', 'PUZZLE-Base', 'PUZZLE']
stages = ['1', '2', '3', '4']
stages_label = ['Stage1 (Gen.)', 'Stage2 (Infer.)', 'Stage3 (Train.)', 'Comm.']


fig, axs = plt.subplots(2, 1, figsize=(10, 6))
index = np.arange(len(target_labels))
bar_width = 0.3
opacity = 1
colors = ['#B4D8E9', '#228CC1', '#BEE39C', '#3CAC3A']

spacing = -0.5
index = index * (1 + spacing)

for loc, cluster in enumerate(clusters):
    data = {s: [0.0 for t in target] for s in stages}
    for i, t in enumerate(target):
        pattern = f'{input_dir}/{cluster}/llama_7b_7b_{t}'
        files = glob.glob(f'{pattern}*')
        assert len(files) > 0, f'No files found with pattern {pattern}'
        with open(files[0], 'r') as f:
            lines = f.readlines()
            st = {'1': [], '2': [], '3': [], '4': []}
            for line in lines:
                if t in ['opt', 'base']:
                    if 'epoch: ' in line:
                        # use re grep the pattern "| stage 1(gen.) (ms): 8930.270 | stage 2(infer.) (ms): 3329.310 | stage 3(train.) (ms): 8254.304 | e2e_time (ms): 20513.883"
                        pattern = r"stage (\d+)\((.*?)\).*? \(ms\): ([\d\.]+)"
                        matches = re.findall(pattern, line)
                        for match in matches:
                            stage, description, time_ms = match
                            st[stage].append(float(time_ms)/1000)
                        if t == 'opt':
                            st['1'][-1] -= st['4'][-1]
                    if 'apply shadow model time' in line:
                        time_s = float(line.split(' ')[-1])
                        st['4'].append(time_s)
                elif t in ['dschat']:
                    if "Generation time" in line:
                        gen_time = float(line.split(' ')[2][:-1])
                        exp_time = float(line.split(' ')[-1])
                        st['1'].append(gen_time)
                        st['2'].append(exp_time - gen_time)
                    elif "Training time:" in line:
                        train_time = float(line.split(' ')[-1])
                        st['3'].append(train_time)
                    elif "Gather latency" in line:
                        comm_time = float(line.split(' ')[3].split('=')[-1][:-1])
                        st['4'].append(comm_time)

            # if t == 'dschat':
            #     continue
            if t == 'base':
                st['4'].extend([0,0])
            # print(st)
            average_stages = {s: sum(st[s][1:]) / len(st[s][1:]) for s in stages}
            # print(f'llama_7b_7b_{t}: {average_stages}')
            for s in stages:
                data[s][i] = (average_stages[s])

    axs[loc].text(39, 0.5, f'on {cluster}', fontsize=12, color='black', ha='center')

    left = np.zeros(len(target_labels))
    for i, (s, ts) in enumerate(data.items()):
        bars = axs[loc].barh(index, ts, bar_width, label=stages_label[i], left=left, color=colors[i % len(colors)])
        left += np.array(ts)

        for bar, value in zip(bars, ts):
            text_x = bar.get_x() + bar.get_width() / 2
            text_y = bar.get_y() + bar.get_height() / 2
            axs[loc].text(text_x, text_y, f'{value:.2f}', ha='center', va='center', color='black')

    axs[loc].set_yticks(index)
    axs[loc].set_yticklabels(target_labels)
    # set x range to 45
    axs[loc].set_xlim(0, 45)
    axs[loc].set_xlabel('Execution time (sec.)')
    # ax.set_title('')
    # ax.legend('')
    # place legend outside the plot, in the upper middle
    axs[loc].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=4)

    axs[loc].grid(axis='x', linestyle='-.')

plt.tight_layout()
plt.savefig(f'{output_dir}/figure12.pdf')
plt.show()