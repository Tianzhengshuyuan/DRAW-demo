import json
import random

import matplotlib.pyplot as plt

from myplot import MyPlot

data = json.load(open("scatter.json"))


def datum2color(datum):
    mapping = {
        'SP': 'r',
        'IMP': 'g',
        'Gretch': 'k',
        'VR': 'y',
        'Tyche': 'b'
    }

    for (k, v) in mapping.items():
        if k in datum['name']:
            return v


def datum2marker(datum):
    mapping = {
        'SP': '.', #点
        'IMP': 'o', #圆
        'Gretch': '^', #上三角
        'VR': 'p', #五边形
        'Tyche': 's' #正方形
    }

    for (k, v) in mapping.items():
        if k in datum['name']:
            return v


def datum2size(datum):
    return 500

def post_hook_func(ax, cfg) :
    # plt.gca().set_yticks([ x/100.0 for x in range(0,260,50)])
    plt.ylim(1.30, 1.60)
    plt.xlim(550, 1100)
    #plt.gca().axhline(y=1.0,ls='-',zorder=0.51,color='red')
    # plt.ylim(top=2.6)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    return

color = [
    '#b5ccc4',
    '#f3d27d',
    '#c66e60',
    '#688fc6',
] 
fig_cfg = {
    'type': 'scatter',
    #'title': 'test title',

    # Real data
    'x': [datum['x'] for datum in data],
    'y': [datum['y'] for datum in data],

    # Annotation text
    'anno': [datum['name'] for datum in data],
    'anno_offset_x': 0,
    'anno_offset_y': 0.015,
    'anno_kwargs': {
        'size': 20,
    },

    # Decoration
    'marker': [datum2marker(datum) for datum in data],
    'size': [datum2size(datum) for datum in data],
    'color': color,

    'label': ['IMP', 'Gretch', 'VR', 'Tyche'],
    'post_main_hook': post_hook_func,

    'xlabel_kwargs': {
        'fontsize': 20,
    },
    'ylabel_kwargs': {
        'fontsize': 20,
    },

    # Label 横纵坐标
    'xlabel': 'Storage Overhead (byte)',
    'ylabel': 'Speedup',

    # Misc
    'grid': True,
    'grid_linestyle': '--',

    'tight': True,
    'figsize' : [10,5],

    # Save
    'save_path': 'myscatter.pdf'
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
