import csv
import json
import argparse
import os

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="fig9")
parser.add_argument('--output', type=str, default="EF2")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件

# 初始化结果列表
result = []

# 读取 CSV 文件
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    headers = next(reader)  # 跳过标题行

    # 遍历前13行数据
    for idx, row in enumerate(reader):
        if idx >= 13:  # 只处理前13行数据
            break
        
        # 跳过空行
        if not any(row):
            continue
        
        # 提取数据
        category = row[0].strip()  # 分类名称
        
        # 处理每一列，处理空字符串转换为默认值 (0.0)
        def safe_float(value):
            return float(value.strip('%')) if value.strip() else 0.0
        
        Original = safe_float(row[1])
        TFLite = safe_float(row[2])
        MNN = safe_float(row[3])
        PDLite = safe_float(row[4])
        ONNX = safe_float(row[5])
        ncnn = safe_float(row[6])
        TFLite_GPU = safe_float(row[7])
        TensorRT = safe_float(row[8])
        TensorRT_NPU = safe_float(row[9])
        CANN = safe_float(row[10])
        OV_CPU = safe_float(row[11])
        OV_GPU = safe_float(row[12])
        OV_NPU = safe_float(row[13])
        
        # 构造 JSON 数据
        result.append({
            "category": category,
            "Original": Original,
            "TFLite": TFLite,
            "MNN": MNN,
            "PDLite": PDLite,
            "ONNX": ONNX,
            "ncnn": ncnn,
            "TFLite_GPU": TFLite_GPU,
            "TensorRT": TensorRT,
            "TensorRT_NPU": TensorRT_NPU,
            "CANN": CANN,
            "OV_CPU": OV_CPU,
            "OV_GPU": OV_GPU,
            "OV_NPU": OV_NPU
        })

# 写入 JSON 文件
with open(output_file, mode="w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print(f"Data has been converted and saved to {output_file}")
