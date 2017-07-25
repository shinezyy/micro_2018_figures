import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin

file_names = [
    'ipc_pred.csv',
]
colors = sns.light_palette("grey", n_colors=3, reverse=True).as_hex()

fig, ax = plt.subplots()
ax.set_ylim([-0.4, 0.4])
fig.set_size_inches((14, 6))
rects = []

for i in range(0, len(file_names)):
    file_name = file_names[i]
    df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', file_name), header=0, sep=',')
    # matrix = df.values
    # print headers
    ipc_col = df['QoS prediction error'].values
    pair_col = df.ix[:, 0].values
    # print pair_col
    # print ipc_col
    num_pairs = len(df.values)
    ind = np.arange(num_pairs)
    width = 0.6
    row = ipc_col

    print(file_name, np.mean(np.abs(row)))
    rects.append(ax.bar(ind + width*i, row, width, color=str(colors[i])))

    # add text
    ax.set_ylabel('IPC Prediction Error')
    # ax.set_title('IPC Prediction Error with Shared Branch Predictor VS Private')
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(pair_col)
    plt.setp(xtick_names, rotation=90, fontsize=12)

    # percentage
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

# ax.legend([x[0] for x in rects], ('Shared BP', 'Partitioned BP'), fontsize='small')

file_name = pjoin('.\\fig\\', file_names[0].rstrip('.txt').replace('_', '-'))
plt.tight_layout()
plt.savefig(file_name.replace('.csv', '.eps'), format='eps')
plt.savefig(file_name.replace('.csv', '.png'))

plt.show()
