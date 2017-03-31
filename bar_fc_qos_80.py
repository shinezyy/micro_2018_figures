import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin

file_names = [
    'qos80_part_all_fc.txt',
]
colors = np.arange(0, 1, 1.0/2)
expected_qos = 0.8

fig, ax = plt.subplots()
# fig.set_size_inches((14, 6))
rects = []


file_name = file_names[0]
df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=None, sep=',')
matrix = df.values
# print headers
# print matrix
num_pairs = len(matrix)
ind = np.arange(num_pairs)
width = 0.4

for i in range(0, 2):
    row = matrix[:, 2+i]
    # print row
    print file_name, np.mean(np.abs(row))
    rects.append(ax.bar(ind + width*i, row, width, color=str(colors[i])))

    # add text
    ax.set_ylabel('QoS')
    ax.set_title('Achieved QoS VS Predicted QoS')
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(matrix[:, 0] + '\n' + matrix[:, 1])
    plt.setp(xtick_names, rotation=45, fontsize=9)

    # percentage
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

ax.set_ylim([0, 1])
ax.legend([x[0] for x in rects], ('Achieved QoS', 'Predicted QoS'),
          fontsize='small')
ax.plot([-width, num_pairs - width], [expected_qos, expected_qos], "k--")
file_name = pjoin('.\\fig\\', file_names[0].rstrip('.txt').replace('_', '-'))
plt.tight_layout()
plt.savefig(file_name+'.eps', format='eps')
plt.savefig(file_name+'.png')

plt.show()
