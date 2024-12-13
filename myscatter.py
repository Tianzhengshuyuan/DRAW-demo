import json
import random
import os
import matplotlib.pyplot as plt

from myplot import MyPlot

data_file = os.path.join("json", "scatter.json")   # 动态设置 JSON 文件
data = json.load(open(data_file))

save_path = os.path.join("pdf", "myscatter.pdf") 


def datum2color(datum):
    mapping = {
        'SP': 'r',
        'IMP': 'g', #green
        'Gretch': 'k', #black
        'VR': 'y', #yello
        'Tyche': 'b' #blue
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
    return 500

def post_hook_func(ax, cfg) :
    # plt.gca().set_yticks([ x/100.0 for x in range(0,260,50)])
    plt.ylim(1.30, 1.60) #设置横轴的范围
    plt.xlim(550, 1100)  #设置纵轴的范围
    
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

    'label': ['IMP', 'Gretch', 'VR', 'Tyche'],
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
    'xlabel': 'Storage Overhead (byte)',
    'ylabel': 'Speedup',

    # Misc
    'grid': True,
    'grid_linestyle': '--',

    #整张图纸的大小
    'tight': True,
    'figsize' : [10,5],

    # Save
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
