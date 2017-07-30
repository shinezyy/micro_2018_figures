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


def get_stat(df: pd.DataFrame):
    qos_mean = np.mean(df['QoS_0'])
    print(qos_mean)
    max = np.max(df['QoS_0']) - qos_mean
    min = qos_mean - np.min(df['QoS_0'])
    overall_qos_mean = np.mean(df['overall QoS'])
    return qos_mean, max, min, overall_qos_mean


def draw(df: pd.DataFrame, color: str, hatch: str,
         ax: matplotlib.axes.SubplotBase,
         ):
    qos_mean = np.mean(df['QoS_0'])
    # qos_std = np.std(df['QoS_0'])
    max = np.max(df['QoS_0']) - qos_mean
    min = qos_mean - np.min(df['QoS_0'])
    overall_qos_mean = np.mean(df['overall QoS'])
    error = np.array([[min], [max]])
    return ax.errorbar(qos_mean, overall_qos_mean, xerr=error, color=color,
                       fmt=hatch, capsize=2)


files = {
    'LPF 85':
        'diff_qos\\dq_85.csv',
    'LPF 90':
        'diff_qos\\dq_90.csv',
    'LPF 95':
        'diff_qos\\dq_95.csv',
    'LPF- 90':
        'diff_qos\\dq_90_fc.csv',
    'Intel':
        'diff_qos\\dq_90_intel.csv',
}

iter_num = 0
colors = sns.light_palette("Black", n_colors=2, reverse=True).as_hex()
rects_list = []
hatchs = [
    '-o',
    '^',
    'x'
]
fig, ax = plt.subplots()

# <editor-fold desc="Axes range">
ax.set_ylim([1.1, 1.6])
ax.set_xlim([0.4, 1])
# ax.spines['bottom'].set_position('zero')
# ax.spines['left'].set_position('zero')
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# </editor-fold>

qoss = [0]*3
maxs = [0]*3
mins = [0]*3
overalls = [0]*3

for f in files:
    df = read_raw(files[f])
    qoss[iter_num], maxs[iter_num], mins[iter_num], \
        overalls[iter_num] = get_stat(df)
    iter_num += 1
    if iter_num == 3:
        break

ret = ax.errorbar(qoss, overalls, xerr=np.array([mins, maxs]), color=colors[0],
            fmt=hatchs[0], capsize=2)
rects_list.append(ret)

df = read_raw(files['LPF- 90'])
q, x, i, o = get_stat(df)
ret = ax.errorbar(q, o, xerr=np.array([[i], [x]]), color=colors[0],
                  fmt=hatchs[1], capsize=2)
rects_list.append(ret)

df = read_raw(files['Intel'])
q, x, i, o = get_stat(df)
ret = ax.errorbar(q, o, xerr=np.array([[i], [x]]), color=colors[0],
                  fmt=hatchs[2], capsize=2)
rects_list.append(ret)


ax.legend([x[0] for x in rects_list],
          ['LPF', 'LPF-', 'Intel'], fontsize='x-small', ncol=5)
ax.set_ylabel('Overall QoS')
ax.set_xlabel('HPT QoS')
plt.tight_layout()

outfile_name = 'fig\\qos-throughput'
plt.savefig(outfile_name+'.eps', format='eps')
plt.savefig(outfile_name+'.png')

plt.show()

