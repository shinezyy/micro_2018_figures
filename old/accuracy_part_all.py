import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin
from matplotlib.colors import ListedColormap

file_name = 'pred_ipc_error.csv'
df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=None, sep=',')
headers = df.values[0]
matrix = df.values[1:]
# print headers
# print matrix
num_hpt = len(matrix)
num_lpt = len(headers) - 1
ind = np.arange(num_hpt)
colors = np.arange(0, 1, 1.0/num_lpt)
# colors = sns.color_palette("cubehelix", num_lpt).as_hex()

width = 0.09

fig, ax = plt.subplots()
rects = []
for i in range(0, num_lpt):
    row = matrix[:, i + 1]
    print row
    rects.append(ax.bar(ind + width*i, row, width, color=str(colors[i])))

# add text
ax.set_ylabel('Prediction Error')
ax.set_title('Prediction Error by HPT and LPT')
ax.set_xticks(ind + width * 5)
ax.set_xticklabels(matrix[:, 0])

# percentage
vals = ax.get_yticks()
ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

# ax.set_ylim([0, 1])

ax.legend([x[0] for x in rects], ['LPT:' + x for x in headers[1:]],
          fontsize='small')

file_name = pjoin('.\\fig\\', file_name.rstrip('.csv').replace('_', '-'))
plt.savefig(file_name+'.eps', format='eps')
plt.savefig(file_name+'.png')

plt.show()
