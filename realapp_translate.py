import csv
import json
import argparse
import os

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="realapp")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{input_name}.json")   # 动态设置 json文件


# 初始化结果列表
result = []

# 读取 CSV 文件
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)
    # 跳过第一行标题
    headers = next(reader)
    
    # 遍历每一行数据
    for row in reader:
        # 跳过空行
        if not any(row):
            continue
        
        # 提取数据
        category = row[0].strip()  # 分类名称

        etbench = float(row[1])
        app = float(row[2])

        # 构造 JSON 数据
        result.append({
            "category": category,
            "etbench": etbench,
            "app": app
        })

# 写入 JSON 文件
with open(output_file, mode="w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print(f"Data has been converted and saved to {output_file}")
