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
画图的命令如下，ymin和ymax是y轴的显示范围，输出的pdf在pdf文件夹内
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

5. 画fig7、fig8、fig10，先处理原始.csv数据，使用命令
```bash
python fig7a_translate.py --input=fig7a --output=CpuFp16Speedup
python fig7b_translate.py --input=fig7b --output=A78Fp16Speedup 
python fig7b_translate.py --input=fig7c --output=A55Fp16Speedup 

python fig8a_translate.py --input=fig8a --output=GpuFp16Speedup
python fig7b_translate.py --input=fig8b --output=G610Fp16Speedup 
python fig7b_translate.py --input=fig8c --output=A660GFp16Speedup 

python fig8a_translate.py --input=fig10a --output=CpuInt8Speedup
python fig7b_translate.py --input=fig10b --output=A78Int8Speedup
python fig7b_translate.py --input=fig10c --output=A55Int8Speedup
```
画图7、8、10a，使用下面的命令
```bash
python fig78a_speedup.py --input=CpuFp16Speedup --ymin=1.3 --ymax=1.9 --fig=fig7a
python fig78a_speedup.py --input=GpuFp16Speedup --ymin=1.1 --ymax=1.8 --fig=fig8a
python fig78a_speedup.py --input=CpuInt8Speedup --ymin=0.5 --ymax=1.04 --fig=fig10a
 
```
如果要把fig7的b和c画到一起，先手动复制fig7b和fig7c的数据到fig7bc.json，然后使用下面的命令画图
```bash
python fig7bc_one.py --input=fig7bc --ymin=1.0 --ymax=2.0 --number=7 --error=1.05
```
同理，对于fig8b、c，手动复制fig8b和fig8c的数据到fig8bc.json，然后使用下面的命令画图
```bash
python fig7bc_one.py --input=fig8bc --ymin=0.8 --ymax=1.7 --number=8 --error=0.845
```
对于fig10b、c，手动复制fig10b和fig10c的数据到fig10bc.json，然后使用下面的命令画图
```bash
python fig7bc_one.py --input=fig10bc --ymin=0.4 --ymax=1.2 --number=10 --error=0.44
```

6. 处理fig9原始数据：
```bash
python fig9_translate_upper.py --input=fig9 --output=Int8Accuracy_upper
python fig9_translate_lower.py --input=fig9 --output=Int8Accuracy_lower
```
画图
```bash
python fig9_accuracy.py --input=Int8Accuracy_upper --ymin=0 --ymax=82 
python fig9_accuracy_blank.py --input=Int8Accuracy_lower --ymin=0 --ymax=82
```

7. 画fig12，使用下面的命令：
```bash
python fig12a_bestengine.py
python fig12b_bestengine.py
```

8. 画fig13，先使用下面的命令从.csv转换到.json
```bash
python fig13_translate.py --input=fig13a --output=A55_acc_lat
python fig13_translate.py --input=fig13b --output=M1_acc_lat
python fig13_translate.py --input=fig13c --output=lioncove_acc_lat
python fig13_translate.py --input=fig13d --output=A660G_acc_lat
python fig13_translate.py --input=fig13e --output=M1_GPU_acc_lat
python fig13_translate.py --input=fig13f --output=lunar_GPU_acc_lat
python fig13_translate.py --input=fig13g --output=AIP_NPU_acc_lat
python fig13_translate.py --input=fig13h --output=M1_NPU_acc_lat
python fig13_translate.py --input=fig13i --output=lunar_NPU_acc_lat
```
接下来画图，使用命令
```bash
python fig13_acc_lat.py --input=A55_acc_lat --ymin=68 --ymax=83 --xmin=30 --xmax=250
python fig13_acc_lat.py --input=M1_acc_lat --ymin=68 --ymax=83 --xmin=3 --xmax=25
python fig13_acc_lat.py --input=lioncove_acc_lat --ymin=68 --ymax=83 --xmin=4 --xmax=32
python fig13_acc_lat.py --input=A660G_acc_lat --ymin=68 --ymax=83 --xmin=4 --xmax=45
python fig13_acc_lat.py --input=M1_GPU_acc_lat --ymin=68 --ymax=83 --xmin=3 --xmax=8
python fig13_acc_lat.py --input=lunar_GPU_acc_lat --ymin=68 --ymax=83 --xmin=0.5 --xmax=3.3
python fig13_acc_lat.py --input=AIP_NPU_acc_lat --ymin=68 --ymax=83 --xmin=3 --xmax=22
python fig13_acc_lat.py --input=M1_NPU_acc_lat --ymin=65 --ymax=83 --xmin=0.4 --xmax=3
python fig13_acc_lat.py --input=lunar_NPU_acc_lat --ymin=68 --ymax=83 --xmin=1 --xmax=9.5
```

9. 画fig11
```bash
python fig11_translate.py 
python fig11_speedup.py --input=GpuInt8Speedup --ymin=0.45 --ymax=1.6
```

10. 画fig14，先处理原始数据
```bash
python fig14_translate_tflite_bar.py
python fig14_translate_tflite_line.py
python fig14_translate_mnn_bar.py
python fig14_translate_mnn_line.py
python fig14_translate_onnx_bar.py
python fig14_translate_onnx_line.py
```
然后画三合一大图
```bash
python fig14_compiler.py
```
11. 画timm-models-a和timm-models-b:
```bash
python timm-models_translate.py
python timm-models_translate.py --input=timm-models-b --output=timm-models-b
```
```bash
python timm-models.py --ymin=0 --ymax=15
python timm-models.py --input=timm-models-b --ymin=0 --ymax=9
```

12. 画sub-3W-power-efficiency和sub-20W-power-efficiency
```bash
python power_efficiency_translate.py
python power_efficiency_translate.py --input=sub-20W-power-efficiency --output=sub-20W-power-efficiency
```
```bash
python power_efficiency.py
python power_efficiency.py --input=sub-20W-power-efficiency --ymin=0 --ymax=20 --twenty=1
```

13. 画realapp
```bash
python realapp_translate.py
python realapp.py 
```