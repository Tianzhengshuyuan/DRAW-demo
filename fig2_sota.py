import json
import random
import os
import matplotlib.pyplot as plt

from myplot import MyPlot

data_file = os.path.join("json", "sota.json")   # 动态设置 JSON 文件
save_path = os.path.join("pdf", "sota.pdf") 

data = json.load(open(data_file))

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

def datum2size(datum):
    #图中每个散点的大小
    return datum['size'] * 150

def post_hook_func(ax, cfg) :
    plt.ylim(68, 84) #设置横轴的范围
    plt.xlim(0.1, 2.1)  #设置纵轴的范围
    ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1]) #手动设置横轴的刻度

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    return


fig_cfg = {
    'name': 'sota',
    'type': 'scatter',

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
    # 'marker': [datum2marker(datum) for datum in data],
    'size': [datum2size(datum) for datum in data],
    'color': [datum2color(datum) for datum in data],
    'scatter_alpha': 0.7,

    'legend': True,  # 启用图例

    
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
    'save_path': save_path
}

if __name__ == '__main__':
    plot = MyPlot(fig_cfg)
    plot.plot()
