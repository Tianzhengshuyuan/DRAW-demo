import csv
import json
import argparse
import os

# 命令行参数解析
parser = argparse.ArgumentParser(description="Generate plots for specific model data")
parser.add_argument('--input', type=str, default="fig3a")
parser.add_argument('--output', type=str, default="fig3a")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件

# 输入和输出文件路径
csv_file = "data.csv"  # 替换为你的 CSV 文件名
json_file = "data.json"  # 输出 JSON 文件名

# 初始化结果列表
result = []

# 读取 CSV 文件并处理数据
with open(input_file, mode="r", encoding="utf-8") as file:
    reader = csv.reader(file)

    for row in reader:
        # 跳过空行
        if not any(row):
            continue

        name = row[0]
        x = float(row[1])
        y = float(row[2])
        model = row[3]

        # 拼接 name
        if model == "CNN":
            name = f"CNN-{name}"

        # 构造 JSON 数据
        result.append({
            "name": name,
            "x": x,
            "y": y
        })

# 写入 JSON 文件
with open(output_file, mode="w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print(f"Data has been processed and saved to {output_file}")
