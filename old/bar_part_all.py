import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin

file_name = 'pred_ipc_error_part_all_64_spec3.txt'
df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=None, sep=',')
matrix = df.values
# print headers
# print matrix
num_pairs = len(matrix)
ind = np.arange(num_pairs)
width = 1

fig, ax = plt.subplots()
fig.set_size_inches((15, 7))
rects = []

row = matrix[:, 2]
print row
print np.mean(np.abs(row))
rects.append(ax.bar(ind, row, width*0.5, color='purple'))

# add text
ax.set_ylabel('Prediction Error')
ax.set_title('Prediction Error with All Resource Partitioned')
ax.set_xticks(ind)
xtick_names = ax.set_xticklabels(matrix[:, 0] + '\n' + matrix[:, 1])
plt.setp(xtick_names, rotation=45, fontsize=9)

# percentage
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

# ax.set_ylim([0, 1])

file_name = pjoin('.\\fig\\', file_name.rstrip('.txt').replace('_', '-'))
plt.savefig(file_name+'.eps', format='eps')
plt.savefig(file_name+'.png')

plt.show()
