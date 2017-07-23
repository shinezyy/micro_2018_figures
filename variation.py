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


def add_qos_to_dict(benchmark, d, qos):
    if benchmark in d:
        d[benchmark].append(qos)
    else:
        d[benchmark] = [qos]


def get_qos_dict(filename):
    df = read_raw(filename)
    d = {}
    for index, row in df.iterrows():
        b0, b1 = index.split('_')
        qos0 = row.loc['QoS_0']
        # qos1 = row.loc['QoS_1']
        add_qos_to_dict(b0, d, qos0)
        # add_qos_to_dict(b1, d, qos1)
    return d


def get_avg_std(x):
    return np.mean(x), np.std(x)


def proc_dict(d):
    df = pd.DataFrame.from_dict(d, orient='index')
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    return df


def draw(df: pd.DataFrame, width: float,
         ax: matplotlib.axes.SubplotBase,
         offset: float, color: str):
    x_labels = tuple(df.index)
    means = tuple(df['mean'])
    stds = tuple(df['std'])
    ind = np.arange(len(x_labels))

    rects = ax.bar(ind + offset, means, width, color=color, yerr=stds)
    ax.set_ylabel('Normalized IPC')
    ax.set_title('Normalized IPC by benchmarks')
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(x_labels)

    return rects


fig, ax = plt.subplots()
ax.set_ylim([0, 1.2])

df_dict = {}

df_dict['comp_core_comp_cache'] =\
    get_qos_dict('variation\\big_core_comp_cache_comp_core_7x7.csv')

df_dict['part_core_part_cache'] = \
    get_qos_dict('variation\\big_core_part_cache_part_core_7x7.csv')

iter_num = 0
width = 0.35
colors = sns.light_palette("grey", n_colors=3, reverse=True).as_hex()
rects_list = []

for k in df_dict:
    df = proc_dict(df_dict[k])
    rects = draw(df=df, width=width, ax=ax,
                 offset=iter_num*width, color=colors[iter_num])
    rects_list.append(rects)
    iter_num += 1

ax.legend([x[0] for x in rects_list],
          [k for k in df_dict], fontsize='small')

plt.tight_layout()
plt.show()

