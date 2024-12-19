import json
import argparse
import os
import pandas as pd
import csv


# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="timm-models-a")
parser.add_argument('--output', type=str, default="timm-models-a")

# 解析命令行参数
args = parser.parse_args()

input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件


# 读取 CSV 并转换为 JSON 格式
data = []
with open(input_file, mode='r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    header = next(csv_reader)  # 读取表头
    ranges = header[1:]  # 第一行中去掉第一列，提取范围名称

    
    # 转置CSV内容，按范围生成JSON数据
    rows = list(csv_reader)
    for col_index, range_name in enumerate(ranges):
        entry = {"name": range_name}  # 每一项的 "name"
        row_count = 0
        for row in rows:
            row_count += 1
            if row_count > 2:
                break
            version = row[0]  # 版本名，例如 v0.9.12
            value = int(row[col_index + 1])  # 对应范围的值
            entry[version] = value  # 填入版本数据
        data.append(entry)

# 写入 JSON 文件
with open(output_file, mode='w', encoding='utf-8') as json_file:
    json.dump(data, json_file, indent=4, ensure_ascii=False)

print(f"CSV 文件已成功转换为 JSON 文件：{output_file}")