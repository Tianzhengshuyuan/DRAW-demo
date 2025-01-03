import functools
import math
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import MultipleLocator, FuncFormatter

from myutil import try_idx, y2btm, y2normalized, find_range_bound


class MyPlot:
    # @formatter:off
    # @staticmethod
    # 检查配置字典 cfg 是否包含指定的键。
    # 如果任何一个键缺失，函数将直接返回 None，不会执行原函数。
    def expect(*keys):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                cfg = args[1] if len(args) >= 2 else self.cfg
                for key in keys:
                    if key not in cfg:
                        return
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    # @staticmethod
    def expectTrue(*keys):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                cfg = args[1] if len(args) >= 2 else self.cfg
                for key in keys:
                    if key not in cfg:
                        return
                    if not cfg[key]:
                        return
                return func(self, *args, **kwargs)
            return wrapper
        return decorator

    # 用于在函数前后加入hook，例如pre_main_hook,post_main_hook在cfg中有设置
    # @staticmethod
    def hook(name):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                ax = args[0] if len(args) >= 2 else plt.gca()
                cfg = args[1] if len(args) >= 2 else self.cfg
                def invoke_cfg_hook(hook_name):
                    from inspect import signature
                    def dummy(ax, cfg):
                        pass
                    hook_func = cfg.get(hook_name, dummy)
                    hook_nargs = len(signature(hook_func).parameters)
                    assert hook_nargs in [0, 2]
                    if hook_nargs == 2:
                        hook_func(ax, cfg)
                    else:
                        hook_func()

                invoke_cfg_hook(f"pre_{name}_hook")
                ret = func(self, *args, **kwargs)
                invoke_cfg_hook(f"post_{name}_hook")
                return ret
            return wrapper
        return decorator
    # @formatter:on

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.fig = None

    # 设置图片的大小和title
    def setup_figure(self):
        self.fig = plt.figure(figsize=self.cfg.get('figsize', None))
        if 'title' in self.cfg:
            plt.title(self.cfg['title'])

    def draw_label(self, ax, cfg):
        if 'xlabel' in cfg:
            xlabel_kwargs = cfg.get("xlabel_kwargs", {})
            ax.set_xlabel(cfg['xlabel'], **xlabel_kwargs)
        if 'ylabel' in cfg:
            ylabel_kwargs = cfg.get("ylabel_kwargs", {})
            ax.set_ylabel(cfg['ylabel'], **ylabel_kwargs)

    def plot(self):
        self.setup_figure()
        # 是否存在键subplot且其值为true
        if 'subplot' in self.cfg and self.cfg['subplot']:
            # cfgs = self.cfg['subplot_cfgs']
            nrows = len(self.cfg['subplot_cfgs'])  # 子图的数量
            print(nrows)
            self.fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=self.cfg['figsize'])
            plt.subplots_adjust(hspace=0.01)  # 调整子图之间的间距

            # 如果只有一个子图，axes 不是列表，需要转换为列表
            if nrows == 1:
                axes = [axes]
                
            # 遍历子图配置，逐个绘制
            for ax, cfg in zip(axes, self.cfg['subplot_cfgs']):
                self.draw_plot(ax, cfg)
        else:
            cfgs = [self.cfg]
            for cfg in cfgs:
                if 'gridspec_kwargs' in self.cfg:
                    # 进一步检查cfg字典中是否包含pos键
                    assert 'pos' in cfg
                    ax = axes[cfg['pos']]
                else:
                    gs = [cfg['grid_spec']] if 'grid_spec' in cfg else []
                    ax = self.fig.add_subplot(*gs)
                self.draw_plot(ax, cfg)

        self.setup_tight()
        self.save()
        self.show()

    def draw_plot(self, ax, cfg):
        self.draw_main(ax, cfg)
        # anno是散点图中每个点旁边的注释
        # self.draw_anno(ax, cfg)
        self.draw_grid(ax, cfg)
        # 如果x轴需要分组
        self.draw_xgroup(ax, cfg)
        
        self.draw_label(ax, cfg)

    # 根据type的类型，选择调用的方法，并传入参数 ax和cfg
    @hook("main")
    def draw_main(self, ax, cfg):
        mapping = {
            'scatter': self.draw_scatter,
            'linebar': self.draw_linebar,
            'pie': self.draw_pie,
            'groupbar': self.draw_groupbar,
            'groupbar_timm': self.draw_groupbar_timm,
            'groupbar_app': self.draw_groupbar_app,
            'groupbar_pe': self.draw_groupbar_pe,
            'groupstackbar': self.draw_groupstackbar,
            'grouplinebar': self.draw_grouplinebar,
            'stackbar': self.draw_stackbar,
            'groupbar_speedup': self.draw_groupbar_speedup,
            'groupbar_speedup_gpu': self.draw_groupbar_speedup_gpu,
            'groupbar_one': self.draw_groupbar_one,            
            'groupbar_accuracy': self.draw_groupbar_accuracy,
            'groupbar_accuracy_blank': self.draw_groupbar_accuracy_blank
        }
        mapping.get(cfg['type'], lambda self: None)(ax, cfg)

    @expectTrue("grid")
    @hook("grid")
    def draw_grid(self, ax, cfg):
        kwargs = cfg.get('grid_kwargs', {})
        if 'grid_linestyle' in cfg:
            kwargs['linestyle'] = cfg['grid_linestyle']
        if 'grid_below' in cfg and cfg['grid_below']:
            # 设置轴线在网格下方
            ax.set_axisbelow(True)
        if 'axis' in cfg:
            kwargs['axis'] = cfg['axis']
        if 'grid_dark_zero' in cfg:
            ax.axhline(0, color='#b0b0b0', linewidth=1.5, linestyle='-', zorder=1)  # 横线
            ax.axvline(0, color='#b0b0b0', linewidth=1.5, linestyle='-', zorder=1)  # 竖线
        
        # 使用 ax.grid 方法绘制网格，传入 cfg['grid'] 和 kwargs
        ax.grid(cfg['grid'], **kwargs, zorder=1)

    @expectTrue("legend")
    def draw_legend(self, ax, cfg):
        kwargs = cfg.get('legend_kwargs', {})
        ax.legend(**kwargs)


    # 设置 Matplotlib 图表的布局为紧凑模式
    @expectTrue("tight")
    @hook("tight")
    def setup_tight(self):
        plt.tight_layout()
    # 添加一个辅助函数，标注超过范围的数值
    def annotate_bars(self, ax, bars, ymax, text_color, x_offset, y_offset):
        for bar in bars:
            height = bar.get_height()

            if height > ymax:  # 超过Y轴最大值时才显示
                ax.text(
                    bar.get_x() + x_offset,  # X位置
                    ymax - y_offset,  # Y位置，略高于最大值
                    f'{height:.1f}',  # 显示数值，保留两位小数
                    ha='center', va='bottom', fontsize=12, color=text_color
                )
    @expect('x', 'y')
    def draw_scatter(self, ax, cfg):
        # `marker` cannot be a list of str
        # plot one point a time, to work around
        for (idx, (x, y)) in enumerate(zip(cfg['x'], cfg['y'])):
            kwargs = {}
            if 'marker' in cfg:
                kwargs['marker'] = cfg['marker'][idx]
            if 'color' in cfg:
                kwargs['c'] = cfg['color'][idx]
            if 'size' in cfg:
                kwargs['s'] = cfg['size'][idx]
            if 'scatter_alpha' in cfg:  # 检查是否有透明度配置
                kwargs['alpha'] = cfg['scatter_alpha']  # 透明度设置

            ax.scatter(x, y, **kwargs, zorder=2)
            if 'name' in cfg :
                if cfg['name'] == 'sota':
                    EF2 = ax.scatter(10000, 10000, s = 80, c = '#4285f4')
                    SWF = ax.scatter(10000, 10000, s = 80, c = '#ea4335')
                    EMO = ax.scatter(10000, 10000, s = 80, c = '#fbbc04')
                    ENX = ax.scatter(10000, 10000, s = 80, c = '#34a853')
                    MV2 = ax.scatter(10000, 10000, s = 80, c = '#ff6d01')
                    MV  = ax.scatter(10000, 10000, s = 80, c = '#46bdc6')
                    LVT = ax.scatter(10000, 10000, s = 80, c = '#ff00ff')
                    CNN = ax.scatter(10000, 10000, s = 80, c = '#999999')
                    ax.legend((EF2, SWF, EMO, ENX, MV2, MV, LVT, CNN), ('EF2', 'SWF', 'EMO', 'ENX', 'MV2', 'MV', 'LVT', 'CNN'), fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=8, frameon=False,
                              handletextpad=0.2, columnspacing=0.8)
                elif 'PCA' in cfg['name']:
                    EF2 = ax.scatter(10000, 10000, s = 80, c = '#4285f4', marker = 'D')
                    SWF = ax.scatter(10000, 10000, s = 80, c = '#ea4335', marker = 'v')
                    EMO = ax.scatter(10000, 10000, s = 80, c = '#fbbc04', marker = '^')
                    ENX = ax.scatter(10000, 10000, s = 80, c = '#34a853', marker = 'p')
                    MV2 = ax.scatter(10000, 10000, s = 80, c = '#ff6d01', marker = 's')
                    MV  = ax.scatter(10000, 10000, s = 80, c = '#46bdc6', marker = 'd')
                    LVT = ax.scatter(10000, 10000, s = 80, c = '#ff00ff', marker = 'h')
                    CNN = ax.scatter(10000, 10000, s = 80, c = '#999999', marker = 'o')
                    ax.legend((EF2, SWF, EMO, ENX, MV2, MV, LVT, CNN), ('EF2', 'SWF', 'EMO', 'ENX', 'MV2', 'MV', 'LVT', 'CNN'),  
                              fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=8, frameon=False,
                              handletextpad=0.2, columnspacing=0.8)

    @expect('x', 'yaxes')
    def draw_linebar(self, ax, cfg):
        ax1 = ax
        #x是['1.Berti', '1.SPP', '2.MLOP', '2.IPCP']
        x = cfg['x']
        bar_cols = 0
        # yaxidx为0、1、2....
        for (yaxidx, yax) in enumerate(cfg['yaxes']):
            if yax['type'] in ('bar', 'normalized_bar'):
                bar_cols += 1

        bar_cols_idx = 0
        for (yaxidx, yax) in enumerate(cfg['yaxes']):
            ax = ax1
            if 'side' in yax and yax['side'] == 'right':
                ax = ax1.twinx()
            self.draw_yax(ax, yax, x, bar_cols=bar_cols, bar_cols_idx=bar_cols_idx)
            if yax['type'] in ('bar', 'normalized_bar'):
                bar_cols_idx += 1.
        # 设置当前活动的坐标轴为 ax1。sca是 set_current_axis的缩写，用于指定后续绘图操作将作用于哪个坐标轴
        plt.sca(ax1)

   
    def draw_groupbar(self, ax, cfg):
        # X轴位置
        # print("draw_groupbar")
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        top1 = cfg['top1']
        top1_div20 = cfg['top1_div20']
        top1_div50 = cfg['top1_div50']
        # 绘制每组柱子
        ax.bar(x - bar_width, top1, width=bar_width, label='Top-1', color='#4285f4', zorder=2, edgecolor='black')
        ax.bar(x, top1_div20, width=bar_width, label='Top-1 div20', color='#ea4335', zorder=2, edgecolor='black')
        ax.bar(x + bar_width, top1_div50, width=bar_width, label='Top-1 div50', color='#fbbc04', zorder=2, edgecolor='black')

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=15)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        # 设置Y轴范围
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)

        # 添加图例
        ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=13, frameon=False,
                  columnspacing=0.8)

    def draw_grouplinebar(self, ax, cfg):
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        A55_bar = cfg['A55_bar']
        A76_bar = cfg['A76_bar']
        M1P_bar = cfg['M1P_bar']

        A55_line = cfg['A55_line']
        A76_line = cfg['A76_line']
        M1P_line = cfg['M1P_line']
        
        # 绘制每组柱子
        bar1 = ax.bar(x - bar_width, A55_bar, width=bar_width, label='A55', color='#4285f4', zorder=2, edgecolor='black')
        bar2 = ax.bar(x, A76_bar, width=bar_width, label='A76', color='#ea4335', zorder=2, edgecolor='black')
        bar3 = ax.bar(x + bar_width, M1P_bar, width=bar_width, label='M1P', color='#fbbc04', zorder=2, edgecolor='black')
        # 绘制折线图
        line1, = ax.plot(x, A55_line, linestyle='--', dashes=(2,3), color='#4285f4', marker='o', markersize=10, label='A55', linewidth=2, zorder=3)
        line2, = ax.plot(x, A76_line, linestyle='--', dashes=(2,3), color='#ea4335', marker='^', markersize=10, label='A76', linewidth=2, zorder=3)
        line3, = ax.plot(x, M1P_line, linestyle='--', dashes=(2,3), color='#fbbc04', marker='*', markersize=15, label='M1P', linewidth=2, zorder=3)

        ymin = cfg['ymin']
        ymax = cfg['ymax']

        # 设置Y轴范围
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        ax.set_xlim(-0.5, 7.5)
        
        ax.axhline(1.0, color='#b0b0b0', linewidth=1, linestyle='--', zorder=1)
        
        if cfg['first'] == True:
            # 添加图例
            # 添加图例：第一行（折线图）
            legend1 = ax.legend(handles=[line1, line2, line3], fontsize=13, loc='upper center', 
                                bbox_to_anchor=(0.4, 1.16), ncol=3, frameon=False, columnspacing=0.8)

            # 添加图例：第二行（柱状图）
            legend2 = ax.legend(handles=[bar1, bar2, bar3], fontsize=13, loc='upper center', 
                                bbox_to_anchor=(0.4, 1.11), ncol=3, frameon=False, columnspacing=0.8)
            
            # 保证legend1不被覆盖
            ax.add_artist(legend1)
            
            # 添加注释文本
            ax.text(0.57, 1.075, "for torch.compile", transform=ax.transAxes, fontsize=13, ha='left', va='center')
            ax.text(0.57, 1.025, "for TVM", transform=ax.transAxes, fontsize=13, ha='left', va='center')
            
        if cfg['last'] == True:
            # 设置X轴刻度和分类名称
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=15)
        else:
            ax.set_xticks([])  # 不显示X轴的刻度
            ax.set_xticklabels([])  # 不显示X轴的标签
            
    def draw_groupbar_speedup(self, ax, cfg):
        # X轴位置
        # print("draw_groupbar")
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度
        if 'bar_width' in cfg:
            bar_width = cfg['bar_width']

        TFLite = cfg['TFLite']
        MNN = cfg['MNN']
        
        ymin = cfg['ymin']
        ymax = cfg['ymax']

        if 'PDLite' in cfg:
            PDLite = cfg['PDLite']
            ncnn = cfg['ncnn']
            
            bars1 = ax.bar(x - bar_width * 1.5, TFLite, width=bar_width, label='TFLite', color='#4285f4', zorder=2, edgecolor='black')
            bars2 = ax.bar(x - bar_width * 0.5, MNN, width=bar_width, label='MNN', color='#ea4335', zorder=2, edgecolor='black')            
            bars3 = ax.bar(x + bar_width * 0.5, PDLite, width=bar_width, label='PDLite', color='#fabc04', zorder=2, edgecolor='black')            
            bars4 = ax.bar(x + bar_width * 1.5, ncnn, width=bar_width, label='ncnn', color='#33a852', zorder=2, edgecolor='black')            
            
            # 标注条形上的数值
            self.annotate_bars(ax, bars1, ymax, '#4285f4',0,0)
            self.annotate_bars(ax, bars2, ymax, '#ea4335',0,0)
            self.annotate_bars(ax, bars3, ymax, '#fabc04',0,0)
            self.annotate_bars(ax, bars4, ymax, '#33a852',0,0)
            # 添加图例
            ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False)

            # ax.set_xlim(-1.3,7)
        else:
            # 绘制每组柱子
            bars1 = ax.bar(x - bar_width/2, TFLite, width=bar_width, label='TFLite', color='#4285f4', zorder=2, edgecolor='black')
            bars2 = ax.bar(x + bar_width/2, MNN, width=bar_width, label='MNN', color='#ea4335', zorder=2, edgecolor='black')
            
            self.annotate_bars(ax, bars1, ymax, '#4285f4',0.5,0.03)
            self.annotate_bars(ax, bars2, ymax, '#ea4335',0.5,0.03) 
             
            for i, value in enumerate(TFLite):
                if math.isnan(value) :# 如果值为负，显示标志
                    ax.text(x[i] - bar_width * 0.5, 0.525, 'x', ha='center', va='top', fontsize=17, color='#4285f4', zorder=3, fontweight='bold')
            for i, value in enumerate(MNN):
                if math.isnan(value) :# 如果值为负，显示标志
                    ax.text(x[i] + bar_width * 0.5, 0.525, 'x', ha='center', va='top', fontsize=17, color='#ea4335', zorder=3,fontweight='bold') 
            ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False)
        
        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=14)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        # 设置Y轴范围
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        ax.axhline(1.0, color='#b0b0b0', linewidth=1, linestyle='--', zorder=1)


    def draw_groupbar_timm(self, ax, cfg):
        # X轴位置
        # print("draw_groupbar")
        name = cfg['name']
        x = np.arange(len(name))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        name = cfg['name']
        v0_9 = cfg['v0_9']
        v1_0 = cfg['v1_0']
        
        ax_right = ax.twinx()
        
        left_array = np.array(v0_9)
        right_array = np.array(v1_0)
        
        scale_factor = left_array.max() / right_array.max() 

        # 绘制每组柱子
        ax.bar(x - bar_width * 0.5, v0_9/scale_factor, width=bar_width, label='v0.9.12', color='#4285f4', zorder=2, edgecolor='black')
        ax_right.bar(x + bar_width * 0.5, v1_0, width=bar_width, label='v1.0.12 inc.', color='#ea4335', zorder=2, edgecolor='black')

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(name, fontsize=15)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        # 设置右侧Y轴范围
        ax_right.set_ylim(ymin, ymax)
        ax_right.set_yticklabels(ax_right.get_yticklabels(), fontsize=15)
        
        # 设置左侧Y轴（Instruction）
        ax.set_ylim(0,left_array.max() / scale_factor * 1.1)
        ax.tick_params(axis='y', labelsize=15) 
        # 自定义右侧Y轴刻度标签
        def format_instruction_ticks(tick, pos):
            value = int(tick * scale_factor)  # 恢复为原始数据
            return str(value)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_instruction_ticks))
        ax_right.set_ylabel('#models', fontsize=15)
        # 添加图例
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = ax_right.get_legend_handles_labels()

        ax.legend(left_handles + right_handles, left_labels + right_labels, fontsize=15, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False)

    def draw_groupbar_timm(self, ax, cfg):
        # X轴位置
        # print("draw_groupbar")
        name = cfg['name']
        x = np.arange(len(name))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        name = cfg['name']
        v0_9 = cfg['v0_9']
        v1_0 = cfg['v1_0']
        
        ax_right = ax.twinx()
        
        left_array = np.array(v0_9)
        right_array = np.array(v1_0)
        
        scale_factor = left_array.max() / right_array.max() 

        # 绘制每组柱子
        ax.bar(x - bar_width * 0.5, v0_9/scale_factor, width=bar_width, label='v0.9.12', color='#4285f4', zorder=2, edgecolor='black')
        ax_right.bar(x + bar_width * 0.5, v1_0, width=bar_width, label='v1.0.12 inc.', color='#ea4335', zorder=2, edgecolor='black')

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(name, fontsize=15)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        # 设置右侧Y轴范围
        ax_right.set_ylim(ymin, ymax)
        ax_right.set_yticklabels(ax_right.get_yticklabels(), fontsize=15)
        
        # 设置左侧Y轴（Instruction）
        ax.set_ylim(0,left_array.max() / scale_factor * 1.1)
        ax.tick_params(axis='y', labelsize=15) 
        # 自定义右侧Y轴刻度标签
        def format_instruction_ticks(tick, pos):
            value = int(tick * scale_factor)  # 恢复为原始数据
            return str(value)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_instruction_ticks))
        ax_right.set_ylabel('#models', fontsize=15)
        # 添加图例
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = ax_right.get_legend_handles_labels()

        ax.legend(left_handles + right_handles, left_labels + right_labels, fontsize=15, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.1), frameon=False)

    def draw_groupbar_pe(self, ax, cfg):
        name = cfg['name']
        x = np.arange(len(name))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        name = cfg['name']
        perf = np.array(cfg['perf'])
        effi = np.array(cfg['effi'])
        
        ax_right = ax.twinx()
        
        left_array = np.array(perf)
        right_array = np.array(effi)

        scale_factor = int(left_array.max() / right_array.max())
        
        ymin = cfg['ymin']
        ymax = cfg['ymax']
        # 绘制每组柱子
        bars1 = ax.bar(x - bar_width * 0.5, perf/scale_factor, width=bar_width, label='Performance', color='#4285f4', zorder=2, edgecolor='black')
        bars2 = ax_right.bar(x + bar_width * 0.5, effi, width=bar_width, label='Performance/W', color='#ea4335', zorder=2, edgecolor='black')

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        if cfg['twenty'] == 1:
            ax.set_xlim(-0.5, 14.5)
            ax.set_xticklabels(name, fontsize=7)  
            self.annotate_bars(ax, bars1, left_array.max()/scale_factor, '#4285f4', -0.45, -38)
            self.annotate_bars(ax, bars2, ymax , '#ea4335', 0.75, -38)       
        else:
            self.annotate_bars(ax, bars1, left_array.max()/scale_factor, '#4285f4', -0.45, -20)
            self.annotate_bars(ax, bars2, ymax , '#ea4335', 0.75, -20)
            ax.set_xlim(-0.5, 12.5)
            ax.set_xticklabels(name, fontsize=10)
            
        # 设置右侧Y轴范围
        ax_right.set_ylim(ymin, ymax)
        
        ax_right.autoscale(enable=False) 
        ax_right.set_yticklabels(ax_right.get_yticklabels(), fontsize=12)
        print(ax_right.get_yticklabels())
        
        # 设置左侧Y轴（Instruction）
        ax.set_ylim(0,left_array.max() / scale_factor * 1.1)
        ax.yaxis.set_major_locator(MultipleLocator(5)) 
        ax.tick_params(axis='y', labelsize=12) 
        # 自定义左侧Y轴刻度标签
        def format_instruction_ticks(tick, pos):
            value = int(tick * scale_factor)  # 恢复为原始数据
            return str(value)

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_instruction_ticks))

        # 添加图例
        left_handles, left_labels = ax.get_legend_handles_labels()
        right_handles, right_labels = ax_right.get_legend_handles_labels()

        ax.legend(left_handles + right_handles, left_labels + right_labels, fontsize=12, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.16), frameon=False)
        
        if cfg['twenty'] == 0:
            ax.add_line(lines.Line2D([-0.5, -0.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([1.5, 1.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([3.5, 3.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([5.5, 5.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([7.5, 7.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([10.5, 10.5], [-9, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([12.5, 12.5], [-9, 0], color='black', linewidth=1, clip_on=False))

            ax.text(0.075, -0.17, "MIX2", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.234, -0.17, "S20P", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.388, -0.17, "K40P", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.541, -0.17, "AC2P", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.733, -0.17, "EG2", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.925, -0.17, "AI PRO", transform=ax.transAxes, fontsize=12, ha='center', va='center')
        else:
            ax.add_line(lines.Line2D([-0.5, -0.5], [-10, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([3.5, 3.5], [-10, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([7.5, 7.5], [-10, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([11.5, 11.5], [-10, 0], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([14.5, 14.5], [-10, 0], color='black', linewidth=1, clip_on=False))

            ax.text(0.135, -0.15, "MTL", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.400, -0.15, "LNL", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.670, -0.15, "M1", transform=ax.transAxes, fontsize=12, ha='center', va='center')
            ax.text(0.900, -0.15, "ORIN", transform=ax.transAxes, fontsize=12, ha='center', va='center')
        print(ax_right.get_ylim())
    def draw_groupbar_app(self, ax, cfg):
        # X轴位置
        # print("draw_groupbar")
        category = cfg['category']
        x = np.arange(len(category))  # 分类的索引
        bar_width = 0.25  # 每个柱子的宽度

        etbench = cfg['etbench']
        app = cfg['app']
        

        # 绘制每组柱子
        ax.bar(x - bar_width * 0.5, etbench, width=bar_width, label='ETBench', color='#4285f4', zorder=2, edgecolor='black')
        ax.bar(x + bar_width * 0.5, app, width=bar_width, label='Application', color='#ea4335', zorder=2, edgecolor='black')

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(category, fontsize=15)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
    
        
        # 设置左侧Y轴（Instruction）
        ax.set_ylim(ymin, ymax)
        ax.tick_params(axis='y', labelsize=15) 

        ax.legend(fontsize=15, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)

    def draw_groupbar_accuracy(self, ax, cfg):
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.06  # 每个柱子的宽度
        
        Original = cfg['Original']
        TFLite = cfg['TFLite']  
        MNN = cfg['MNN']
        PDLite = cfg['PDLite']
        ONNX = cfg['ONNX']
        ncnn = cfg['ncnn']
        TFLite_GPU = cfg['TFLite_GPU']
        TensorRT = cfg['TensorRT']
        TensorRT_NPU = cfg['TensorRT_NPU']
        CANN = cfg['CANN']
        OV_CPU = cfg['OV_CPU']
        OV_GPU = cfg['OV_GPU']
        OV_NPU = cfg['OV_NPU']

        bars = [
            (Original, '#999999', 'Original', -6, ''),
            (TFLite, '#4185f3', 'TFLite', -5, '//'),
            (TFLite_GPU, '#4185f3', 'TFLite(GPU)', -4, '\\\\'),
            (MNN, '#ea4234', 'MNN', -3, ''),
            (PDLite, '#fabc04', 'PDLite', -2, ''),
            (ONNX, '#ff9632', 'ONNXRT', -1, ''),
            (ncnn, '#33a852', 'ncnn', -0, ''),
            (TensorRT, '#BA55D3', 'TensorRT', 1, '//'),
            (TensorRT_NPU, '#BA55D3', 'TensorRT(NPU)', 2, '\\\\'),
            (CANN, '#D38DD6', 'CANN', 3, ''),
            (OV_CPU, '#87CEFA', 'OpenVINO(CPU)', 4, '//'),
            (OV_GPU, '#87CEFA', 'OpenVINO(GPU)', 5, '..'),
            (OV_NPU, '#87CEFA', 'OpenVINO(NPU)', 6, '\\\\'),
        ]
        
        for data, color, label, offset, hatch in bars:
            for i, value in enumerate(data):
                if value < 0:  # 如果值为负，显示标志
                    ax.text(
                        x[i] + bar_width * offset, 4.5,  
                        'x', ha='center', va='top', fontsize=17, color=color, zorder=3
                    )
            ax.bar(x + bar_width * offset, data, width=bar_width, label=label, color=color, zorder=2, edgecolor='black', hatch=hatch)

        ax.add_line(lines.Line2D([-0.5, -0.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([2.5, 2.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([5.5, 5.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([8.5, 8.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([11.5, 11.5], [-6, 0], color='black', linewidth=1, clip_on=False))
         
        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=20)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        ax.grid(axis='y', linestyle='--')
        # 设置X、Y轴范围
        ax.set_xlim(-0.5,11.5)
        ax.set_ylim(ymin, ymax)

        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        order = [0,7,1,8,2,9,3,10,4,11,5,12,6]
        ax.legend([handles[idx] for idx in order], [labels[idx] for idx in order],fontsize=20, loc='upper center', bbox_to_anchor=(0.48, 1.21), ncol=7, frameon=False)

    def draw_groupbar_accuracy_blank(self, ax, cfg):
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        x = np.array([0,1,2,3,4,5,6,7,8,10,11])
        bar_width = 0.06  # 每个柱子的宽度
        
        Original = cfg['Original']
        TFLite = cfg['TFLite']  
        MNN = cfg['MNN']
        PDLite = cfg['PDLite']
        ONNX = cfg['ONNX']
        ncnn = cfg['ncnn']
        TFLite_GPU = cfg['TFLite_GPU']
        TensorRT = cfg['TensorRT']
        TensorRT_NPU = cfg['TensorRT_NPU']
        CANN = cfg['CANN']
        OV_CPU = cfg['OV_CPU']
        OV_GPU = cfg['OV_GPU']
        OV_NPU = cfg['OV_NPU']

        bars = [
            (Original, '#999999', 'Original', -6, ''),
            (TFLite, '#4185f3', 'TFLite', -5, '//'),
            (TFLite_GPU, '#4185f3', 'TFLite(GPU)', -4, '\\\\'),
            (MNN, '#ea4234', 'MNN', -3, ''),
            (PDLite, '#fabc04', 'PDLite', -2, ''),
            (ONNX, '#ff9632', 'ONNXRT', -1, ''),
            (ncnn, '#33a852', 'ncnn', -0, ''),
            (TensorRT, '#BA55D3', 'TensorRT', 1, '//'),
            (TensorRT_NPU, '#BA55D3', 'TensorRT(NPU)', 2, '\\\\'),
            (CANN, '#D38DD6', 'CANN', 3, ''),
            (OV_CPU, '#87CEFA', 'OpenVINO(CPU)', 4, '//'),
            (OV_GPU, '#87CEFA', 'OpenVINO(GPU)', 5, '..'),
            (OV_NPU, '#87CEFA', 'OpenVINO(NPU)', 6, '\\\\'),
        ]
        
        for data, color, label, offset, hatch in bars:
            for i, value in enumerate(data):
                if value < 0:  # 如果值为负，显示标志
                    ax.text(
                        x[i] + bar_width * offset, 4.5,  
                        'x', ha='center', va='top', fontsize=17, color=color, zorder=3
                    )
            ax.bar(x + bar_width * offset, data, width=bar_width, label=label, color=color, zorder=2, edgecolor='black', hatch=hatch)

        ax.add_line(lines.Line2D([-0.5, -0.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([2.5, 2.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([5.5, 5.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([8.5, 8.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([9.5, 9.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([10.5, 10.5], [-6, 0], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([11.5, 11.5], [-6, 0], color='black', linewidth=1, clip_on=False))
         
        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=20)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        ax.grid(axis='y', linestyle='--')
        # 设置X、Y轴范围
        ax.set_xlim(-0.5,11.5)
        ax.set_ylim(ymin, ymax)

    def draw_groupbar_speedup_gpu(self, ax, cfg):
        # X轴位置
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.06  # 每个柱子的宽度
        if 'bar_width' in cfg:
            bar_width = cfg['bar_width']

        G31 = cfg['G31']    
        G52 = cfg['G52']
        G610 = cfg['G610']
        G77 = cfg['G77']
        A630G = cfg['A630G']
        A660G = cfg['A660G']
        A740G = cfg['A740G']
        AMP = cfg['AMP']
        MTL_GPU = cfg['MTL_GPU']
        LNL_GPU = cfg['LNL_GPU']
        ORIN_NPU = cfg['ORIN_NPU']
        AIP_NPU = cfg['AIP_NPU']
        MTL_NPU = cfg['MTL_NPU']
        LNL_NPU = cfg['LNL_NPU']
        
        
        ymin = cfg['ymin']
        ymax = cfg['ymax']
    
        #ff6c00 橙  #ea4234 红  #fabc04 黄
        #33a852 绿  #4185f3 蓝  #9900ff 紫
        colors_large_gap = [
            "#ea4234",  
            "#ea4234",  
            "#ea4234",  
            "#ff6c00",  
            "#ff6c00",  
            "#ff6c00",  
            "#ff6c00",  
            "#fabc04",  
            "#fabc04",  
            "#33a852",  
            "#33a852",  
            "#4185f3",  
            "#4185f3",   
            "#9900ff"  
        ]
        bars1 = ax.bar(x - bar_width * 6.5, G31, hatch='//', width=bar_width, label='G31', color=colors_large_gap[0], zorder=2, edgecolor='black', alpha=0.8)
        bars2 = ax.bar(x - bar_width * 5.5, G52, hatch='\\\\', width=bar_width, label='G52', color=colors_large_gap[1], zorder=2, edgecolor='black', alpha=0.8)            
        bars3 = ax.bar(x - bar_width * 4.5, G610, hatch='..', width=bar_width, label='G610', color=colors_large_gap[2], zorder=2, edgecolor='black', alpha=0.8)            
        bars4 = ax.bar(x - bar_width * 3.5, G77, hatch='//', width=bar_width, label='G77', color=colors_large_gap[3], zorder=2, edgecolor='black', alpha=0.8)            
        bars5 = ax.bar(x - bar_width * 2.5, A630G, hatch='\\\\', width=bar_width, label='A630G', color=colors_large_gap[4], zorder=2, edgecolor='black', alpha=0.8)            
        bars6 = ax.bar(x - bar_width * 1.5, A660G, hatch='..', width=bar_width, label='A660G', color=colors_large_gap[5], zorder=2, edgecolor='black', alpha=0.8)            
        bars7 = ax.bar(x - bar_width * 0.5, A740G, width=bar_width, label='A740G', color=colors_large_gap[6], zorder=2, edgecolor='black', alpha=0.8)            
        bars8 = ax.bar(x + bar_width * 0.5, AMP, hatch='//', width=bar_width, label='AMP', color=colors_large_gap[7], zorder=2, edgecolor='black', alpha=0.8)            
        bars9 = ax.bar(x + bar_width * 1.5, ORIN_NPU, hatch='\\\\', width=bar_width, label='ORIN_NPU', color=colors_large_gap[8], zorder=2, edgecolor='black', alpha=0.8)            
        bars10 = ax.bar(x + bar_width * 2.5, MTL_GPU, hatch='//', width=bar_width, label='MTL_GPU', color=colors_large_gap[9], zorder=2, edgecolor='black', alpha=0.8)            
        bars11 = ax.bar(x + bar_width * 3.5, MTL_NPU, hatch='\\\\', width=bar_width, label='MTL_NPU', color=colors_large_gap[10], zorder=2, edgecolor='black', alpha=0.8)            
        bars12 = ax.bar(x + bar_width * 4.5, LNL_GPU, hatch='//', width=bar_width, label='LNL_GPU', color=colors_large_gap[11], zorder=2, edgecolor='black', alpha=0.8)            
        bars13 = ax.bar(x + bar_width * 5.5, LNL_NPU, hatch='\\\\', width=bar_width, label='LNL_NPU', color=colors_large_gap[12], zorder=2, edgecolor='black', alpha=0.8) 
        bars14 = ax.bar(x + bar_width * 6.5, AIP_NPU,  width=bar_width, label='AIP_NPU', color=colors_large_gap[13], zorder=2, edgecolor='black', alpha=0.8)            

        # 标注条形上的数值，使用 colors_large_gap 对应的颜色
        self.annotate_bars(ax, bars1, ymax, colors_large_gap[0], 0.15, 0.07)
        self.annotate_bars(ax, bars2, ymax, colors_large_gap[1], 0.15, 0.07)
        self.annotate_bars(ax, bars3, ymax, colors_large_gap[2], 0.15, 0.07)
        self.annotate_bars(ax, bars4, ymax, colors_large_gap[3], 0.15, 0.07)
        self.annotate_bars(ax, bars5, ymax, colors_large_gap[4], 0.15, 0.07)
        self.annotate_bars(ax, bars6, ymax, colors_large_gap[5], 0.15, 0.07)
        self.annotate_bars(ax, bars7, ymax, colors_large_gap[6], 0.15, 0.07)
        self.annotate_bars(ax, bars8, ymax, colors_large_gap[7], 0.15, 0.07)
        self.annotate_bars(ax, bars9, ymax, colors_large_gap[8], 0.15, 0.07)
        self.annotate_bars(ax, bars10, ymax, colors_large_gap[9], 0.15, 0.07)
        self.annotate_bars(ax, bars11, ymax, colors_large_gap[10], 0.15, 0.07)
        self.annotate_bars(ax, bars12, ymax, colors_large_gap[11], 0.15, 0.07)
        self.annotate_bars(ax, bars13, ymax, colors_large_gap[12], 0.15, 0.07)
        self.annotate_bars(ax, bars14, ymax, colors_large_gap[13], 0.15, 0.07)
        # 添加图例
        ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=14, frameon=False,
                handletextpad=0.2, handlelength = 1.3, columnspacing=0.2)

        # ax.set_xlim(-1.3,7)

        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=14)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        # 设置Y轴范围
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        ax.set_xlim(-0.5, 6.5)
        
        ax.axhline(1.0, color='#b0b0b0', linewidth=1, linestyle='--', zorder=1)
        
        for i, bar in enumerate([bars1, bars2, bars3, bars4, bars5, bars6, bars7, bars8, bars9, bars10, bars11, bars12, bars13, bars14]):
            for rect in bar:
                if rect.get_height() == -1:
                    ax.text(rect.get_x() + rect.get_width() / 2, 0.48, 'X', color=colors_large_gap[i], ha='center', va='center', fontsize=10)
        
        ax.add_line(lines.Line2D([-0.5, -0.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([0.5, 0.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([1.5, 1.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([2.5, 2.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([3.5, 3.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([4.5, 4.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([5.5, 5.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))
        ax.add_line(lines.Line2D([6.5, 6.5], [0.4, 0.5], color='black', linewidth=1, clip_on=False))

    def draw_groupbar_one(self, ax, cfg):
        # X轴位置
        categories = cfg['categories']
        x = np.arange(len(categories))  # 分类的索引
        bar_width = 0.2  # 每个柱子的宽度

        TFLite = cfg['TFLite']
        MNN = cfg['MNN']
        PDLite = cfg['PDLite']
        ncnn = cfg['ncnn']
        
        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        error = cfg['error']
        ax.axhline(1.0, color='#b0b0b0', linewidth=1, linestyle='--', zorder=1)
        
        #加上叉叉
        for i, value in enumerate(PDLite):
            print(type(value))
            if math.isnan(value) :# 如果值为负，显示标志
                ax.text(x[i] + bar_width * 0.5, error, 'x', ha='center', va='top', fontsize=17, color='#fabc04', zorder=3, fontweight='bold')
        for i, value in enumerate(ncnn):
            print(type(value))
            if math.isnan(value) :# 如果值为负，显示标志
                ax.text(x[i] + bar_width * 1.5, error, 'x', ha='center', va='top', fontsize=17, color='#33a852', zorder=3,fontweight='bold')     
                       
        bars1 = ax.bar(x - bar_width * 1.5, TFLite, width=bar_width, label='TFLite', color='#4285f4', zorder=2, edgecolor='black')
        bars2 = ax.bar(x - bar_width * 0.5, MNN, width=bar_width, label='MNN', color='#ea4335', zorder=2, edgecolor='black')            
        bars3 = ax.bar(x + bar_width * 0.5, PDLite, width=bar_width, label='PDLite', color='#fabc04', zorder=2, edgecolor='black')            
        bars4 = ax.bar(x + bar_width * 1.5, ncnn, width=bar_width, label='ncnn', color='#33a852', zorder=2, edgecolor='black')            
        
        # 标注条形上的数值
        self.annotate_bars(ax, bars1, ymax, '#4285f4', 0.45, 0.04)
        self.annotate_bars(ax, bars2, ymax, '#ea4335', 0.45, 0.04)
        self.annotate_bars(ax, bars3, ymax, '#fabc04', 0.45, 0.04)
        self.annotate_bars(ax, bars4, ymax, '#33a852', 0.45, 0.04)
        # 添加图例
        ax.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)

        ax.set_xlim(-0.8,13.6)
        
        # 设置X轴刻度和分类名称
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=15)

        ymin = cfg['ymin']
        ymax = cfg['ymax']
        
        # 设置Y轴范围
        ax.set_ylim(ymin, ymax)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        
        number = cfg['number']
        if number == 7:
            ax.add_line(lines.Line2D([6.4, 6.4], [0.85, 1], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([-0.8, -0.8], [0.85, 1], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([13.6, 13.6], [0.85, 1], color='black', linewidth=1, clip_on=False))
            ax.text(3, 0.9, "A78", ha='center', va='top', clip_on=False, fontsize=17)  
            ax.text(10, 0.9, "A55", ha='center', va='top', clip_on=False, fontsize=17)  
        elif number == 8:
            ax.add_line(lines.Line2D([6.4, 6.4], [0.66, 0.8], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([-0.8, -0.8], [0.66, 0.8], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([13.6, 13.6], [0.66, 0.8], color='black', linewidth=1, clip_on=False))
            ax.text(3, 0.71, "G610", ha='center', va='top', clip_on=False, fontsize=17)  
            ax.text(10, 0.71, "A660G", ha='center', va='top', clip_on=False, fontsize=17) 
        else:
            ax.add_line(lines.Line2D([6.4, 6.4], [0.28, 0.4], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([-0.8, -0.8], [0.28, 0.4], color='black', linewidth=1, clip_on=False))
            ax.add_line(lines.Line2D([13.6, 13.6], [0.28, 0.4], color='black', linewidth=1, clip_on=False))
            ax.text(3, 0.32, "A78", ha='center', va='top', clip_on=False, fontsize=17)  
            ax.text(10, 0.32, "A55", ha='center', va='top', clip_on=False, fontsize=17)   
    def draw_groupstackbar(self, ax, cfg):
        # 数据
        categories = cfg['categories']
        Latency = cfg['Latency']
        Instruction = cfg['Instruction']
        latency_label=[]
        color1 = cfg['color1']
        color2 = cfg['color2']
        label1 = cfg['label1']
        label2 = cfg['label2']
        
        length = cfg['length']
        width = cfg['width']

        x = np.arange(len(categories))  # X轴位置
        bar_width = 0.28  # 每组宽度

        # 创建图形
        fig, ax_Latency = plt.subplots(figsize=(length, width))

        # 添加第二个Y轴
        ax_Instruction = ax_Latency.twinx()
        
        latency_array = np.array(Latency)
        instruction_array = np.array(Instruction)
        
        scale_factor = instruction_array.max() / latency_array.max() 
        btm1 = y2btm(latency_array)
        btm2 = y2btm(instruction_array)
        
        # 左侧分组堆叠柱 (Latency)
        for (catidx, (cat, bb)) in enumerate(zip(latency_array.T, btm1.T)):
            ax_Latency.bar(x - bar_width / 2, cat, width=bar_width, bottom=bb, label=label1[catidx], color=color1[catidx], edgecolor='black', linestyle='-', alpha=1)

        # 右侧分组堆叠柱 (Instruction)
        for (catidx, (cat, bb)) in enumerate(zip(instruction_array.T, btm2.T)):
            ax_Instruction.bar(x + bar_width / 2, cat/scale_factor, width=bar_width, bottom=bb/scale_factor, label=label2[catidx], color=color2[catidx], edgecolor='black', linestyle='-', alpha=1)


        # 设置X轴标签
        ax_Latency.set_xticks(x)
        ax_Latency.set_xticklabels(categories, fontsize=15)

        # 设置左侧Y轴（Latency）
        ax_Latency.set_ylabel('latency (ms)', fontsize=15)
        ax_Latency.set_ylim(0, latency_array.sum(axis=1).max() * 1.1)
        ax_Latency.tick_params(axis='y', labelsize=15) 

        # 设置右侧Y轴（Instruction）
        ax_Instruction.set_ylabel('Instructions number', fontsize=15)
        ax_Instruction.set_ylim(0,instruction_array.sum(axis=1).max() / scale_factor * 1.1)
        ax_Instruction.tick_params(axis='y', labelsize=15) 
        # 自定义右侧Y轴刻度标签
        def format_instruction_ticks(tick, pos):
            value = int(tick * scale_factor)  # 恢复为原始数据
            if value >= 1e6:
                return f'{value // 1e6}M'  # 百万单位
            elif value >= 1e3:
                return f'{value // 1e3}K'  # 千单位
            else:
                return str(value)

        ax_Instruction.yaxis.set_major_formatter(ticker.FuncFormatter(format_instruction_ticks))
        # 添加图例
        Latency_handles, Latency_labels = ax_Latency.get_legend_handles_labels()
        Instruction_handles, Instruction_labels = ax_Instruction.get_legend_handles_labels()
        order = [3,2,1,0]
        Latency_labels = [Latency_labels[i] for i in order]
        Latency_handles = [Latency_handles[i] for i in order]
        Instruction_labels = [Instruction_labels[i] for i in order]
        Instruction_handles = [Instruction_handles[i] for i in order]
        ax_Latency.legend(Latency_handles + Instruction_handles, Latency_labels + Instruction_labels,  ncol=2,  loc='upper left', fontsize=12)

        # 调整布局
        plt.tight_layout()
        plt.show()

    def draw_stackbar(self, ax, cfg):
        # 数据
        categories = cfg['categories']
        Latency = cfg['Latency']
        latency_label=[]
        color1 = cfg['color1']
        label1 = cfg['label1']
        
        length = cfg['length']
        width = cfg['width']

        x = np.arange(len(categories)) # X轴位置
        bar_width = cfg['bar_width']  # 每组宽度

        # 创建图形
        fig, ax_Latency = plt.subplots(figsize=(length, width))

        latency_array = np.array(Latency)
        
        btm1 = y2btm(latency_array)

        # 堆叠柱 (Latency)
        for (catidx, (cat, bb)) in enumerate(zip(latency_array.T, btm1.T)):
            ax_Latency.bar(x, cat, width=bar_width, bottom=bb, label=label1[catidx], color=color1[catidx], edgecolor='black', linestyle='-', alpha=1)

        # 设置X轴标签
        ax_Latency.set_xticks(x)
        ax_Latency.set_xticklabels(categories, fontsize=12)
        if len(categories) == 2:
            ax_Latency.set_xlim(-0.75,1.75)

        # 设置Y轴（Latency）
        ax_Latency.set_ylabel('latency (ms)', fontsize=14)
        ax_Latency.set_ylim(0, latency_array.sum(axis=1).max() * 1.1)
        ax_Latency.tick_params(axis='y', labelsize=15)

        
        # 添加图例
        if len(color1) == 4:
            order = [3,2,1,0]
        else:
            order = [4,3,2,1,0]
            
        Latency_handles, Latency_labels = ax_Latency.get_legend_handles_labels()
        Latency_labels = [Latency_labels[i] for i in order]
        Latency_handles = [Latency_handles[i] for i in order]
        ax_Latency.legend(Latency_handles, Latency_labels,  loc='upper left', fontsize=12)

        # 调整布局
        plt.tight_layout()
        plt.show()
    @staticmethod
    def draw_line(ax, x, yax, **kwargs):
        y = yax['y']
        for (catidx, cat) in enumerate(y.T):
            kwargs = try_idx(yax.get("line_kwargs", {}), catidx)
            if 'marker' in yax:
                kwargs['marker'] = try_idx(yax['marker'], catidx)
            if 'color' in yax:
                kwargs['c'] = try_idx(yax['color'], catidx)
            if 'linestyle' in yax:
                kwargs['ls'] = try_idx(yax['linestyle'], catidx)
            if 'label' in yax:
                kwargs['label'] = try_idx(yax['label'], catidx)
            ax.plot(x, cat, **kwargs)

    @staticmethod
    def draw_spline_line(ax, x, yax, **kwargs):
        y = yax['y']

        xrange = list(range(len(y)))
        xsmooth = np.linspace(0, len(y) - 1, 100 * (len(y) - 1))

        for (catidx, cat) in enumerate(y.T):
            kwargs = {}
            if 'color' in yax:
                kwargs['c'] = try_idx(yax['color'], catidx)
            if 'linestyle' in yax:
                kwargs['ls'] = try_idx(yax['linestyle'], catidx)
            spline = make_interp_spline(xrange, cat)
            ysmooth = spline(xsmooth)
            ax.plot(xsmooth, ysmooth, **kwargs)

            if 'marker' in yax:
                kwargs['marker'] = yax['marker']
            kwargs['ls'] = ''
            ax.plot(x, cat, **kwargs)

    @staticmethod
    def draw_bar(ax, x, yax, **kwargs):
        # x = ['1.Berti', '1.SPP', '2.MLOP', '2.IPCP']
        # y是yases中的第一组数据
        # bar_cols默认设置为1，bar_cols_idx默认设置为0
        bar_cols = kwargs.get('bar_cols', 1)
        bar_cols_idx = kwargs.get('bar_cols_idx', 0)

        #从kwargs中移除
        kwargs.pop('bar_cols')
        kwargs.pop('bar_cols_idx')

        # y是linebar1.json中的y
        y = yax['y']
        if yax['type'] == 'normalized_bar':
            y = y2normalized(y)

        # y.shape[0] = 4，据此生成xrange = [0,1,2,3]
        xrange = np.arange(y.shape[0])
        group_width = 0.7 #grouped_bar_kwargs.get("group_width", 1.0)
        padding = 0.05 #grouped_bar_kwargs.get("padding", 0.)
        width = (group_width + padding) / (bar_cols)
        xpos = xrange - (group_width + padding) / 2 + (bar_cols_idx + 0.5) * width
        #堆叠bar中每个小bar的起始高度
        #[[ 0.  5.  8.]
        # [ 0.  7. 14.]
        # [ 0.  3.  4.]
        # [ 0.  8. 15.]]
        btm = y2btm(y)
        
        for (catidx, (cat, bb)) in enumerate(zip(y.T, btm.T)):
            # cat是[5 7 3 8]、[3 7 1 7]、[2 0 9 1]
            # bb是[0. 0. 0. 0.]
                 #[5. 7. 3. 8.]
                 #[ 8. 14.  4. 15.]
            if 'color' in yax:
                #kwargs['color']是r,k,y
                kwargs['color'] = try_idx(yax['color'], catidx)
            if 'hatch' in yax:
                kwargs['hatch'] = try_idx(yax['hatch'], catidx)
                kwargs['alpha'] = 0.99
            if 'label' in yax:
                #kwargs['label']是stride、simple、complex
                kwargs['label'] = try_idx(yax['label'], catidx)
            ax.bar(xpos, cat, width=width - padding, bottom=bb, **kwargs)
            # ax.bar(xrange - group_width / 2 + (catidx + 1) * width, cat, width=width - padding, **kwargs)
    @staticmethod
    def draw_grouped_bar(ax, x, yax, **kwargs):
        grouped_bar_kwargs = yax.get("grouped_bar_kwargs", {})

        y = yax['y']

        xrange = np.arange(y.shape[0])
        group_width = grouped_bar_kwargs.get("group_width", 1.0)
        padding = grouped_bar_kwargs.get("padding", 0.)
        width = group_width / (y.shape[1] + 1)

        ax.set_xticks(xrange, x)
        for (catidx, cat) in enumerate(y.T):
            kwargs = {}
            if 'color' in yax:
                kwargs['color'] = try_idx(yax['color'], catidx)
            if 'label' in yax:
                kwargs['label'] = try_idx(yax['label'], catidx)
            xpos = xrange - group_width / 2 + (catidx + 1) * width
            print(xpos)
            # ax.bar(xpos, cat, width=width - padding, **kwargs)

    @staticmethod
    def draw_grouped_line(ax, x, yax, **kwargs):
        grouped_line_kwargs = {'color':'black', 'linestyle':"--", "lw":1}
        grouped_line_kwargs.update( yax.get("grouped_line_kwargs", {}))

        y = yax['y']
        n_cats = y.shape[1]

        xrange = np.arange(y.shape[0])
        group_width = grouped_line_kwargs.get("group_width", 1.0)
        padding = grouped_line_kwargs.get("padding", 0.)
        width = group_width / (n_cats + 1)

        for (grpidx, grp) in enumerate(y):
            xpos = grpidx - group_width / 2 + (np.arange(n_cats) + 1) * width
            ax.plot(xpos, grp, **grouped_line_kwargs)

        ax.set_xticks(xrange, x)
        for (catidx, cat) in enumerate(y.T):
            kwargs = {}
            if 'color' in yax:
                kwargs['color'] = try_idx(yax['color'], catidx)
            if 'label' in yax:
                kwargs['label'] = try_idx(yax['label'], catidx)
            if 'marker' in yax:
                kwargs['marker'] = try_idx(yax['marker'], catidx)
            kwargs['linestyle'] = 'None'
            xpos = xrange - group_width / 2 + (catidx + 1) * width
            ax.plot(xpos, cat, **kwargs)

    @expect('x', 'y')
    def draw_pie(self, ax, cfg):
        x = cfg['x']
        y = cfg['y']
        pie_kwargs = cfg.get("pie_kwargs", {})
        ax.pie(y, labels=x, **pie_kwargs)

    @hook("yax")
    def draw_yax(self, ax, yax, x, **kwargs):
        mapping = {
            'line': MyPlot.draw_line,
            'spline_line': MyPlot.draw_spline_line,
            'bar': MyPlot.draw_bar,
            'normalized_bar': MyPlot.draw_bar,
            'grouped_bar': MyPlot.draw_grouped_bar,
            'grouped_line': MyPlot.draw_grouped_line,
        }
        # 会直接执行对应的函数
        mapping[yax['type']](ax, x, yax, **kwargs)
        if 'axlabel' in yax:
            axlabel_kwargs = yax.get('axlabel_kwargs', {})
            ax.set_ylabel(yax['axlabel'], **axlabel_kwargs)
        if 'grid' in yax:
            self.draw_grid(ax, yax)
        if 'legend' in yax:
            self.draw_legend(ax, yax)

    @expect("anno", 'x', 'y')
    @hook("anno")
    def draw_anno(self, ax, cfg):
        # 如果没有anno_offset_x键的话，offset_x设为0
        offset_x = cfg.get('anno_offset_x', 0)
        offset_y = cfg.get("anno_offset_y", 0)
        for (anno_idx, (anno, x, y)) in enumerate(zip(cfg['anno'], cfg['x'], cfg['y'])):
            kwargs = try_idx(cfg.get("anno_kwargs", {}), anno_idx)
            # 使用 ax.annotate 方法在图表中添加注释，注释的位置为 (x + offset_x, y + offset_y)
            ax.annotate(anno, (x + offset_x, y + offset_y), **kwargs)

    # 将当前的图表保存到配置中指定的路径
    @expect("save_path")
    def save(self):
        # plt.savefig(self.cfg['save_path'])
        plt.savefig(self.cfg['save_path'], bbox_inches='tight', pad_inches=0.02)
    # 显示所有当前创建的绘图窗口
    def show(self):
        plt.show()

    @expectTrue("xgroup")
    def draw_xgroup(self, ax, cfg):
        # 从配置对象 cfg 中获取键 “xgroup_kwargs"的值，如果不存在则返回空字典{}
        kwargs = cfg.get("xgroup_kwargs", {})
        delimiter = kwargs.get('delimiter', '.')
        x_split = [x.split(delimiter) for x in cfg['x']]

        xnum = len(cfg['x'])
        width = 1 / xnum
        
        # 获取x轴的最小值和最大值，并将它们分别赋值给变量 xmin 和 xmax    
        (xmin, xmax) = ax.get_xlim()
        xrange = xmax - xmin

        (ymin, ymax) = ax.get_ylim()
        yrange = ymax - ymin

        xscale = xnum / xrange
        yfactor = kwargs.get('yfactor', 0.5)  # y scaling factor defaults to 0.5
        yoffset = kwargs.get('yoffset', 0.2)
        yscale = yrange / plt.gcf().get_size_inches()[1] * yfactor
        xxlate = - (xmin + 0.5) / xrange
        yxlate = ymin
        xaffine = Affine2D().scale(xscale, yscale).translate(xxlate, yxlate)
        trans = xaffine + ax.get_yaxis_transform()

        x_split_T = [*zip(*x_split)]
        # 关闭 x 轴底部的标签显示
        ax.tick_params(labelbottom=False)  # disable built-in label
        minlvl = kwargs.get('minlevel', 0)
        for (lvl, x_split_lvl) in enumerate(reversed(x_split_T)):
            line_kwargs = try_idx(kwargs.get('line_kwargs', {}), lvl)
            text_kwargs = try_idx(kwargs.get('text_kwargs', {}), lvl)
            ranges, bounds = find_range_bound(x_split_lvl)
            if lvl >= minlvl:
                for b in bounds:
                    pos_x = ((b * width), (b * width))
                    pos_y = (0, (-lvl - yoffset))
                    ax.add_line(Line2D(pos_x, pos_y,
                                       transform=trans, clip_on=False, color="k", **line_kwargs))
            for ((l, r), x) in ranges:
                pos_x = (l + r) / 2 * width
                pos_y = (-lvl - yoffset)
                ax.text(pos_x, pos_y, x,
                        horizontalalignment='center', verticalalignment='top',
                        clip_on=False, transform=trans, **text_kwargs)


if __name__ == '__main__':
    plot = MyPlot({'size': (12, 6)})
    plot.setup_figure()