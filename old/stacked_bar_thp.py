import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random as rd
from os.path import join as pjoin

file_names = [
    'fc_thp.txt',
    'cc_thp.txt',
    'dyn_thp.txt',
]

expected_qos = 0.9
colors = sns.color_palette("Paired").as_hex()[int(expected_qos*10) % 3 * 2 :]


fig, ax = plt.subplots()
# fig.set_size_inches((14, 6))
rects = []



width = 0.3

for i in range(0, 3):
    file_name = file_names[i]
    df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=None, sep=',')
    matrix = df.values
    num_pairs = len(matrix)
    ind = np.arange(num_pairs)


    hpts = matrix[:, 2]
    lpts = matrix[:, 3]
    # print row
    print(file_name, np.mean(np.abs(hpts + lpts)))
    rects.append(ax.bar(ind + width * i, hpts, width, color=colors[i]))
    rects.append(ax.bar(ind + width * i, lpts, width, bottom=hpts, color=colors[i+2]))

    # add text
    ax.set_ylabel('Throughput')
    # ax.set_title('')
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(matrix[:, 0] + '\n' + matrix[:, 1])
    plt.setp(xtick_names, rotation=45, fontsize=9)

ax.legend([x[0] for x in rects], ('HPT IPC (FrontEnd)', 'LPT IPC (FrontEnd)',
                                  'HPT IPC (Combind)', 'LPT IPC (Combind)',
                                  'HPT IPC (Dynamic)', 'LPT IPC (Dynamic)',),
          fontsize='small')

file_name = 'qos90_thp_cmp.txt'
file_name = pjoin('.\\fig\\', file_name.rstrip('.txt').replace('_', '-'))
plt.tight_layout()
plt.savefig(file_name+'.eps', format='eps')
plt.savefig(file_name+'.png')

plt.show()
