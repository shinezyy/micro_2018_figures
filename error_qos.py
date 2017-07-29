import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from os.path import join as pjoin
import sys


def read_raw(filename):
    with open(filename) as f:
        df = pd.read_csv(filepath_or_buffer=f, header=0, sep=',',
                         index_col=0, engine='python')
        return df


def draw(df: pd.DataFrame, color: str, hatch: str,
         ax: matplotlib.axes.SubplotBase,
         ):
    return ax.plot(df['IPC prediction error'], df['QoS_0'] - 0.9, hatch, color=color,)


files = {
    'controlled_core_controlled_cache':
        'qos_48\\big_core_cc_cache_cc_core_6.csv',
    'controlled_fetch_controlled_cache':
        'qos_48\\big_core_cc_cache_fc_core_4.csv',
}

iter_num = 0
width = 0.18
colors = sns.light_palette("Black", n_colors=2, reverse=True).as_hex()
rects_list = []
hatchs = [
    '.',
    '+'
]
fig, ax = plt.subplots()
ax.set_ylim([-0.4, 0.15])
ax.set_xlim([-0.4, 0.4])
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

for f in files:
    df = read_raw(files[f])
    rects_list.append(draw(df=df, color=colors[0], hatch=hatchs[iter_num], ax=ax))
    iter_num += 1

plt.tight_layout()
ax.legend([x[0] for x in rects_list],
          [k for k in files], fontsize='xx-small', ncol=5)

outfile_name = 'fig\\error-qos'
plt.savefig(outfile_name+'.eps', format='eps')
plt.savefig(outfile_name+'.png')

plt.show()

