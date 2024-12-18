import csv
import json
import argparse
import os

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="fig11")
parser.add_argument('--output', type=str, default="GpuInt8Speedup")

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
        if idx >= 7:  # 只处理前7行数据
            break
        
        # 跳过空行
        if not any(row):
            continue
        
        # 提取数据
        category = row[0].strip()  # 分类名称
        
        # 处理每一列，处理空字符串转换为默认值 (0.0)
        def safe_float(value):
            return float(value) if value else -1.0
        
        G31 = safe_float(row[1])
        G52 = safe_float(row[2])
        G610 = safe_float(row[3])
        G77 = safe_float(row[4])
        A630G = safe_float(row[5])
        A660G = safe_float(row[6])
        A740G = safe_float(row[7])
        AMP = safe_float(row[8])
        MTL_GPU = safe_float(row[9])
        LNL_GPU = safe_float(row[10])
        ORIN_NPU = safe_float(row[11])
        AIP_NPU = safe_float(row[12])
        MTL_NPU = safe_float(row[13])
        LNL_NPU = safe_float(row[14])
        
        # 构造 JSON 数据
        result.append({
            "category": category,
            "G31": G31,
            "G52": G52,
            "G610": G610,
            "G77": G77,
            "A630G": A630G,
            "A660G": A660G,
            "A740G": A740G,
            "AMP": AMP,
            "MTL_GPU": MTL_GPU,
            "LNL_GPU": LNL_GPU,
            "ORIN_NPU": ORIN_NPU,
            "AIP_NPU": AIP_NPU,
            "MTL_NPU": MTL_NPU,
            "LNL_NPU": LNL_NPU
        })

# 写入 JSON 文件
with open(output_file, mode="w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print(f"Data has been converted and saved to {output_file}")
