import numpy as np
import os
from myplot import MyPlot

save_path = os.path.join("pdf", "mypie.pdf") 

fig_cfg = {
    "type": "pie",
    "x": ["apple", "orange", "pear", "grape"],
    "y": [0.1, 0.3, 0.4, 0.2],
    "pie_kwargs": {"autopct": "%1.1f%%"}, #小数点位数，1.1显示1位小数，1.5显示5位
    "tight": False,
    "save_path": save_path
}


if __name__ == "__main__":
    plot = MyPlot(fig_cfg)
    plot.plot()
