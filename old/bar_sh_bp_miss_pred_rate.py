import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin

file_names = [
    'mis_pred_rate_share_bp.txt',
    'mis_pred_rate_part_all.txt',
]
colors = sns.light_palette("green", n_colors=3, reverse=True).as_hex()

fig, ax = plt.subplots()
fig.set_size_inches((14, 6))
rects = []

for i in range(0, len(file_names)):
    file_name = file_names[i]
    df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=None, sep=',')
    matrix = df.values
    # print headers
    # print matrix
    num_pairs = len(matrix)
    ind = np.arange(num_pairs)
    width = 0.4


    row = matrix[:, 2]
    # print row
    print file_name, np.mean(row)
    rects.append(ax.bar(ind + width*i, row, width, color=str(colors[i])))

    # add text
    ax.set_ylabel('Branch Miss Prediction Rate')
    ax.set_title('Branch Miss Prediction Rate with Shared Branch Predictor VS Private')
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(matrix[:, 0] + '\n' + matrix[:, 1])
    plt.setp(xtick_names, rotation=45, fontsize=9)

    # percentage
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

# ax.set_ylim([0, 1])
ax.legend([x[0] for x in rects], ('Shared BP', 'Partitioned BP'),
          fontsize='small')

file_name = pjoin('.\\fig\\', file_names[0].rstrip('.txt').replace('_', '-'))
plt.tight_layout()
plt.savefig(file_name+'.eps', format='eps')
plt.savefig(file_name+'.png')

plt.show()
