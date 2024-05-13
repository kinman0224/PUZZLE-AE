import matplotlib.pyplot as plt
import glob
import sys

input_dir=sys.argv[1]
output_dir=sys.argv[2]

clusters = ['orion']
target = ['opt', 'dschat']
target_label = ['PUZZLE', 'DS-Chat']
actor_critic = ['7b_350m', '13b_350m', '33b_7b']

data = {ac: [] for ac in actor_critic}
for c in clusters:
    for t in target:
        for ac in actor_critic:
        # read data from logs with pattern "llama_{ac}_{t}_"
            pattern = f'{input_dir}/llama_{ac}_{t}_'
            files = glob.glob(f'{pattern}*')
            assert len(files) > 0, f'No files found with pattern {pattern}'
            with open(files[0], 'r') as f:
                lines = f.readlines()
                overhead_list = []
                if t == 'opt':
                    for line in lines:
                        if 'shadow model time' in line:
                            overhead = float(line.split(' ')[-1])
                            overhead_list.append(overhead)
                elif t == 'dschat':
                    for line in lines:
                        if 'Gather latency' in line:
                            overhead = float(line.split(' ')[3].split('=')[-1][:-1])
                            overhead_list.append(overhead)
                average_overhead = sum(overhead_list[1:]) / len(overhead_list[1:])
                # print(f'{ac}: {average_overhead}')
                data[ac].append(f"{average_overhead:.2f}")

# use matplotlib plot table
fig, ax = plt.subplots(figsize=(8, 3))
index = range(len(actor_critic))
cell_text = []
for ac in actor_critic:
    cell_text.append(data[ac])

# plot table
# Add headers and a table at the bottom of the axes
header_0 = ax.table(cellText=[['']*len(clusters)],
                     colLabels=clusters,
                     loc='upper center',
                     )
header_0.auto_set_font_size(False)
header_0.set_fontsize(10)
header_0.scale(0.8, 2)

table = ax.table(cellText=cell_text, rowLabels=actor_critic, colLabels=target_label * 2, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(0.8, 2)
plt.axis('off')
plt.savefig(f"{output_dir}/table3.pdf")