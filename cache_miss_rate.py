import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as pjoin

csv_file_name = 'st_miss_rate.csv'
df = pd.read_csv(filepath_or_buffer=pjoin('.\\csv', csv_file_name), header=None, sep=',')
headers = df.values[0]
matrix = df.values[1:]
# print headers
# print matrix
num_targets = len(matrix)
num_benchmarks = len(headers) - 1
ind = np.arange(num_benchmarks)
width = 0.5


rects = []
for i in range(0, num_targets):
    fig, ax = plt.subplots()
    row = matrix[i][1:]
    print row
    rects.append(ax.bar(ind, row, width, color='0'))
    stat = matrix[i][0]
    # add text
    # ax.set_ylabel(stat)
    ax.set_title(stat)
    ax.set_xticks(ind)
    xtick_names = ax.set_xticklabels(headers[1:])
    plt.setp(xtick_names, rotation=25, fontsize=11)

    # percentage
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:3.2f}%'.format(x*100) for x in vals])

    file_name = pjoin('.\\fig\\', csv_file_name.rstrip('.csv').replace('_', '-') + stat.replace(' ', '-'))
    plt.savefig(file_name+'.eps', format='eps')
    plt.savefig(file_name+'.png')


    # plt.show()
