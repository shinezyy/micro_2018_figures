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


def get_qos_dict(filename):
    df_raw = read_raw(filename)
    d = {}
    for index, row in df_raw.iterrows():
        d[index] = row.loc['IPC prediction error']
    df = pd.DataFrame.from_dict(d, orient='index')
    df.loc['mean'] = np.mean(df[0].abs())
    print('mean=', df.loc['mean'])
    return df


def draw(df: pd.DataFrame, width: float,
         ax: matplotlib.axes.SubplotBase,
         offset: float, color: str, set_xtickname: bool):
    x_labels = tuple(df.index)
    ind = np.arange(len(x_labels))
    # print(ind)

    # print(df)
    # sys.exit(0)
    rects = ax.bar(ind + offset, df[0], width, color=color)
    # ax.set_ylabel('Solo IPC prediction error')
    ax.set_title('Solo IPC prediction error', fontsize=8)
    if (set_xtickname):
        ax.set_xticks(ind)
        xtick_names = ax.set_xticklabels(x_labels)
        plt.setp(xtick_names, rotation=90, fontsize=7)

    return rects


fig, ax = plt.subplots()
# ax.set_ylim([0, 2.2])
ax.set_xlim([-0.5, 50])
fig.set_size_inches((6, 3.5))

df_dict = {
    'controlled_core_controlled_cache':
        get_qos_dict('qos_48\\big_core_cc_cache_cc_core_6.csv'),
    'controlled_fetch_controlled_cache':
        get_qos_dict('qos_48\\big_core_cc_cache_fc_core_4.csv'),
}

iter_num = 0
width = 0.4
colors = sns.light_palette("grey", n_colors=6, reverse=True).as_hex()
rects_list = []

for k in df_dict:
    df = df_dict[k]
    rects = draw(df=df, width=width, ax=ax,
                 offset=iter_num*width, color=colors[iter_num],
                 set_xtickname=iter_num == 0)
    rects_list.append(rects)
    iter_num += 1

ax.legend([x[0] for x in rects_list],
          [k for k in df_dict], fontsize='xx-small', ncol=5)

plt.tight_layout()

outfile_name = 'fig\\error'
plt.savefig(outfile_name+'.eps', format='eps')
plt.savefig(outfile_name+'.png')

plt.show()

