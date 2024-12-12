import json
import random

import matplotlib.pyplot as plt

from myplot import MyPlot

data = json.load(open("sota.json"))

def datum2color(datum):
    mapping = {
        'EF2': '#4285f4',
        'SWF': '#ea4335',
        'EMO': '#fbbc04',
        'ENX': '#34a853',
        'MV2': '#ff6d01',
        'MV':  '#46bdc6',
        'LVT': '#ff00ff',
        'CNN': '#999999'
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
    #图中每个散点的大小
    # return 500
    return datum['size'] * 150

def post_hook_func(ax, cfg) :
    # plt.gca().set_yticks([ x/100.0 for x in range(0,260,50)])
    plt.ylim(68, 84) #设置横轴的范围
    plt.xlim(0.1, 2.1)  #设置纵轴的范围
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]) 
    #plt.gca().axhline(y=1.0,ls='-',zorder=0.51,color='red')
    # plt.ylim(top=2.6)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    return

color = [
    '#b5ccc4',#绿
    '#f3d27d',#黄
    '#c66e60',#红
    '#688fc6',#蓝
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
        'size': 10, #每个散点上标注的名字的大小
    },

    # Decoration
    'marker': [datum2marker(datum) for datum in data],
    'size': [datum2size(datum) for datum in data],
    'color': [datum2color(datum) for datum in data],
    'scatter_alpha': 0.7,

    # 'label': ['EF2', 'SWF', 'EMO', 'ENX', 'MV2', 'MV', 'LVT', 'CNN'],  # 图例的类别名称
    'legend': True,  # 启用图例
    'legend_kwargs': {
        'frameon': True
    },
    
    'post_main_hook': post_hook_func,

    #横轴名称(label)的大小
    'xlabel_kwargs': {
        'fontsize': 15,
    },
    #纵轴名称(label)的大小
    'ylabel_kwargs': {
        'fontsize': 15,
    },

    # Label 横纵坐标
    'xlabel': 'FLOPs (G)',
    'ylabel': 'ImageNet-1K Top-1 (%)',

    # Misc
    'grid': True,
    'grid_linestyle': '--',

    #整张图纸的大小
    'tight': True,
    'figsize' : [10,6],

    # Save
    'save_path': 'sota.pdf'
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
