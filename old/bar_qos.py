import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin
import sys

qos = 90

file_names = [
    'big_core_comp_cache_part_core_48.csv',
    'big_core_cc_cache_cc_core_3.csv',
]

legends = [
    'comp cache part core',
    'controlled cache & core',
]

dyn_file = 'dyn.csv'
outfile_name = '.\\fig\\qos-achieved'
expected_qos = float(qos)/100
colors = sns.light_palette("grey", n_colors=4, reverse=True).as_hex()

fig, ax = plt.subplots()
ax.set_ylim([0, 1.2])
fig.set_size_inches((14, 6))
rects = []
width = 0.3

num_pairs = 0

# get pairs with poor performance when no control
df = pd.read_csv(filepath_or_buffer=pjoin('.\\qos_48', dyn_file), header=0, sep=',', index_col=0)
poor_indices = df[abs(df['QoS_0']) < 0.8].index
# print df.loc[poor_indices, :]

for i in range(0, len(file_names)):
    file_name = file_names[i]
    df = pd.read_csv(filepath_or_buffer=pjoin('.\\qos_48', file_name), header=0, sep=',', index_col=0)
    # matrix = df.values
    # print headers
    df = df.loc[poor_indices, :]
    qos_col = df['QoS_0'].values
    num_pairs = len(df.values)
    ind = np.arange(num_pairs)
    row = qos_col

    print(file_name, 'mean:', np.mean(np.abs(row)), 'std:', np.std(row),
          'error:', np.mean(np.abs(row - float(qos)/100)))
    rects.append(ax.bar(ind + width*(i - 1), row, width, color=str(colors[i])))

    # add text
    ax.set_ylabel('Achieved QoS')
    # ax.set_title('IPC Prediction Error with Shared Branch Predictor VS Private')
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(poor_indices.values)
    plt.setp(xtick_names, rotation=90, fontsize=12)

    # percentage
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

ax.legend([x[0] for x in rects], legends, fontsize='small')

# file_name = pjoin('.\\fig\\', file_names[0].rstrip('.txt').replace('_', '-'))

ax.plot([-width, num_pairs - width], [expected_qos, expected_qos], "k--")
plt.tight_layout()
# plt.savefig(outfile_name+'.eps', format='eps')
# plt.savefig(outfile_name+'.png')

plt.show()
