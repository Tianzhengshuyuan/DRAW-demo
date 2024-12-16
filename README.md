# 环境配置
使用下面的指令配置环境
```bash
conda create -yn plot python=3.10
conda activate plot
pip install numpy
pip install matplotlib
pip install scipy
pip install pandas
```
# 基础作图
如果要画折线/柱状图、饼图、散点图，分别使用下面的指令：
```bash
python mylinebar.py
python mypie.py
python myscatter.py
```

# 论文作图
1. 画fig2中的sota.pdf，使用下面的指令，使用的数据在json/sota.json
```bash
python fig2_sota.py
```

2. 画fig5中的EF2.pdf、SWF.pdf、EMO.pdf等，首先使用fig5_translate.py处理原始.csv文件中的数据，输出到.json文件中。命令如下，其中input是输入的.csv文件名，model是要处理的模型
```bash
python fig5_translate.py --input=fig5 --model=EF2
python fig5_translate.py --input=fig5 --model=SWF
python fig5_translate.py --input=fig5 --model=EMO
python fig5_translate.py --input=fig5 --model=ENX 
python fig5_translate.py --input=fig5 --model=MV2
python fig5_translate.py --input=fig5 --model=MV-
python fig5_translate.py --input=fig5 --model=LVT
```
接着使用fig5_accuracy.py画图，注意除EF2外的其他模型，如果不需要ylabel的话，先注释掉fig5_accuracy.py中的这一行
```python
    # 'ylabel': 'Top-1 accuracy (%)',
```
转换的命令如下，ymin和ymax是y轴的显示范围，输出的pdf在pdf文件夹内
```bash
python fig5_accuracy.py --model=EF2 --ymin=75 --ymax=83
python fig5_accuracy.py --model=SWF --ymin=75 --ymax=83
python fig5_accuracy.py --model=EMO --ymin=68 --ymax=80
python fig5_accuracy.py --model=ENX --ymin=70 --ymax=82
python fig5_accuracy.py --model=MV2 --ymin=66 --ymax=79
python fig5_accuracy.py --model=MV-  --ymin=66 --ymax=79
python fig5_accuracy.py --model=LVT --ymin=75 --ymax=81
```

3. 画fig3，首先使用fig3_translate.py处理fig3a.csv文件中的数据，输出到independent_PCA.json文件中，fig3b同理。命令如下：
```bash
python fig3_translate.py --input=fig3a --output=independent_PCA 
python fig3_translate.py --input=fig3b --output=dependent_PCA
```
接着使用fig3a_accuracy.py画图，得到independent_PCA.pdf
```bash
python fig3a_PCA.py
```
使用fig3b_accuracy.py画图，得到dependent_PCA.pdf
```bash
python fig3b_PCA.py
```

4. 画fp16vsint8的指令如下：
如果要画cpu的对比图，使用命令：
```bash
python fp16vsint8_cpu.py
```
如果要画gpu的对比图，使用命令：
```bash
python fp16vsint8_gpu.py
```
如果要画npu的对比图，使用命令：
```bash
python fp16vsint8_npu.py
```

5. 画fig7a、fig7b、fig7c、fig8a、fig8b、fig8c，先处理原始.csv数据，使用命令
```bash
python fig7a_translate.py --input=fig7a --output=CpuFp16Speedup
python fig7b_translate.py --input=fig7b --output=A78Fp16Speedup 
python fig7b_translate.py --input=fig7c --output=A55Fp16Speedup 
python fig8a_translate.py --input=fig8a --output=GpuFp16Speedup
python fig7b_translate.py --input=fig8b --output=G610Fp16Speedup 
python fig7b_translate.py --input=fig8c --output=A660GFp16Speedup 
python fig8a_translate.py --input=fig10a --output=CpuInt8Speedup
python fig7b_translate.py --input=fig10b --output=A78Int8Speedup
python fig7b_translate.py --input=fig10c --output=A510Int8Speedup
```
然后作图，使用命令
```bash
python fig78a_speedup.py --input=CpuFp16Speedup --ymin=1.3 --ymax=2.15
python fig78a_speedup.py --input=GpuFp16Speedup --ymin=1.1 --ymax=1.8
python fig78b_speedup.py --input=A78Fp16Speedup --ymin=1.1 --ymax=2.0
python fig78b_speedup.py --input=A55Fp16Speedup --ymin=1.0 --ymax=2.0
python fig78b_speedup.py --input=G610Fp16Speedup --ymin=1.0 --ymax=1.7
python fig78b_speedup.py --input=A660GFp16Speedup --ymin=0.8 --ymax=1.7
```