# 环境配置
使用下面的指令配置环境
```bash
conda create -yn plot python=3.10
conda activate plot
pip install numpy
pip install matplotlib
pip install scipy
```
# 基础作图
如果要画折线/柱状图、饼图、散点图，分别使用下面的指令：
```bash
python mylinebar.py
python mypie.py
python myscatter.py
```

# 论文作图
1. 要画sota.pdf,使用指令,数据在sota.json：
```bash
python sota.py
```

2. 要画EF2.pdf、SWF.pdf、EMO.pdf等，首先使用translate.py处理原始.csv文件中的数据，命令如下，其中input是输入的.csv文件名，model是要处理的模型：
```bash
python translate.py --input=fig2 --model=EF2
python translate.py --input=fig2 --model=SWF
python translate.py --input=fig2 --model=EMO
python translate.py --input=fig2 --model=ENX 
python translate.py --input=fig2 --model=MV2
python translate.py --input=fig2 --model=MV
python translate.py --input=fig2 --model=LVT
```
接着使用accuracy.py画图，注意除了EF2外的其他模型，如果不需要ylabel的话，先注释掉accuracy.py中的这一行：
```python
    # 'ylabel': 'Top-1 accuracy (%)',
```
转换的命令如下，ymin和ymax是y轴的显示范围：
```bash
python accuracy.py --model=EF2 --ymin=75 --ymax=83
python accuracy.py --model=SWF --ymin=75 --ymax=83
python accuracy.py --model=EMO --ymin=68 --ymax=80
python accuracy.py --model=ENX --ymin=70 --ymax=82
python accuracy.py --model=MV2 --ymin=66 --ymax=79
python accuracy.py --model=MV  --ymin=66 --ymax=79
python accuracy.py --model=LVT --ymin=75 --ymax=81
```

