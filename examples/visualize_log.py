import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlrd


def visualize_log(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if figsize is None:
        figsize = (15., 5. * len(keys))
    f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
    for idx, key in enumerate(keys):
        axarr[idx].plot(episodes, data[key])
        axarr[idx].set_ylabel(key)
    plt.xlabel('episodes')
    plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

def visualize_meanQ(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    # ['duration', 'episode_reward', 'loss', 'mean_absolute_error', 'mean_q', 'nb_episode_steps', 'nb_steps']
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if figsize is None:
        figsize = (15., len(keys))
    plt.figure(figsize=figsize)
    plt.plot(episodes, data['mean_q'])
    plt.ylabel('Mean Q-values')
    plt.xlabel('Episodes')
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

def visualize_reward(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    # ['duration', 'episode_reward', 'loss', 'mean_absolute_error', 'mean_q', 'nb_episode_steps', 'nb_steps']
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if figsize is None:
        figsize = (15., len(keys))
    plt.figure(figsize=figsize)
    df = pd.DataFrame(data['episode_reward'])
    plt.plot(df[0], 'lightblue', df[0].rolling(8).mean(), 'b')
    plt.ylabel('Episode Rewards')
    plt.xlabel('Episodes')
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def xlrd_open(filename):
    workbook = xlrd.open_workbook(filename)
    first_sheet = workbook.sheet_by_index(0)

    reward = [first_sheet.cell_value(i, 1) for i in range(first_sheet.nrows)]

    return reward

def visualize_dp(filename, figsize=None, output=None):
    reward = xlrd_open(filename)
    optimum = reward.index(max(reward))
    plt.plot(reward, marker='.', markersize=8)
    plt.vlines(optimum, 0, max(reward), colors='orange', linestyles='dashed')
    plt.hlines(max(reward), 0, optimum, colors='orange', linestyles='dashed')
    # plt.annotate(r'$(%s, %s)$'%(optimum, max(reward)), xy=(optimum, max(reward)), xycoords='data',
    #              xytext=(3.5, 9.3), fontsize=14, color='darkblue', fontweight='bold')
    plt.annotate(r'$(%s, %s)$' % (optimum, max(reward)), xy=(optimum, max(reward)), xycoords='data',
                 xytext=(8.2, 8.2), fontsize=14, color='darkblue', fontweight='bold')
    plt.axis([0, len(reward), 0, 10])
    plt.xlabel('Episode Index', fontdict={'size':12.5})
    plt.ylabel('Episode Rewards', fontdict={'size':12.5})
    plt.title('Deep PILCO', fontdict={'size':16, 'color':'midnightblue', 'weight':'bold', 'style':'italic'})

    if output is None:
        plt.show()
    else:
        plt.savefig(output)


def visualize_dpall(output=None):
    iter1_50 = '/home/jingyi/Desktop/Charts/1v1/10/dp-iter1-50-050503.xls'
    iter1_51 = '/home/jingyi/Desktop/Charts/1v1/10/dp-iter1-51-050504.xls'
    iter10_15 = '/home/jingyi/Desktop/Charts/1v1/10/dp-iter10-15-050502.xls'

    reward1_50 = xlrd_open(iter1_50)
    reward1_51 = xlrd_open(iter1_51)
    reward10_15 = xlrd_open(iter10_15)

    plt.plot(reward1_50, label='Goal:9, Rollout:1')
    plt.plot(reward1_51, label='Goal:7, Rollout:1')
    plt.plot(reward10_15, label='Goal:9, Rollout:10')
    plt.axis([0, 50, 0, 10])
    plt.xlabel('Episode Index', fontdict={'size': 12.5})
    plt.ylabel('Episode Rewards', fontdict={'size': 12.5})
    plt.legend(loc='best')

    if output is None:
        plt.show()
    else:
        plt.savefig(output)


parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, default=None, help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.

visualize_log(args.filename, output=args.output, figsize=args.figsize)

# visualize_log(args.filename, output=args.output+'fig.pdf', figsize=args.figsize)
# visualize_meanQ(args.filename, output=args.output+'meanQ.pdf', figsize=args.figsize)
# visualize_reward(args.filename, output=args.output+'reward.pdf', figsize=args.figsize)

# visualize_dp(args.filename, output=args.output, figsize=args.figsize)
# visualize_dpall(output=args.output)