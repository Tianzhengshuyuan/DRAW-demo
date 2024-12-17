import csv
import json
import os
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="fig13a")
parser.add_argument('--output', type=str, default="M1P_acc_lat")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件

# 读取CSV文件并转换为JSON
result = []
with open(input_file, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)  # 跳过标题行
    next(reader)  # 跳过第二行说明行
    
    count = 0  # 计数器用于给 "name" 添加索引
    
    for row in reader:
        if not any(row):  # 跳过空行
            continue
        
        name = f"{row[0]}-s{count}"  # 构造name字段 (例如: EF2-s0)
        # 检查并处理缺失值，给出默认值
        try:
            size = float(row[1]) if row[1] else -100.0  # 如果缺失，设置为0.0
        except ValueError:
            size = -100.0  # 无效数据时设置为0.0
        
        try:
            y = float(row[2].strip('%')) if row[2] else -100.0  # 如果缺失，设置为0.0
        except ValueError:
            y = -100.0
        
        try:
            x = float(row[3]) if row[3] else -100.0  # 如果缺失，设置为0.0
        except ValueError:
            x = -100.0

        result.append({
            "name": name,
            "x": x,
            "y": y,
            "size": size
        })
        count += 1  # 递增索引

# 将结果写入JSON文件
with open(output_file, "w", encoding="utf-8") as json_file:
    json.dump(result, json_file, indent=4, ensure_ascii=False)

print(f"转换完成！JSON 文件已保存为 {output_file}")
