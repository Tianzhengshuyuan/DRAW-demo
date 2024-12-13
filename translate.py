import csv
import json
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser(description="Generate plots for specific model data")
parser.add_argument('--input', type=str, default="fig2" , help="The origin csv file name.")
parser.add_argument('--model', type=str, default="EF2",  help="The name of the model of which the data will be processed.")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input
model_name = args.model

# 动态加载 JSON 数据和设置保存路径
input_file = f"{input_name}.csv"  # 动态设置 csv文件
output_file = f"{model_name}.json"  # 动态设置 json文件


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
        if category.startswith(model_name):  # 仅处理以 EF2 开头的行
            top1 = float(row[1].strip('%'))
            top1_div20 = float(row[2].strip('%'))
            top1_div50 = float(row[3].strip('%'))

            # 构造 JSON 数据
            result.append({
                "category": category,
                "top1": top1,
                "top1_div20": top1_div20,
                "top1_div50": top1_div50
            })

# 写入 JSON 文件
with open(output_file, mode="w", encoding="utf-8") as file:
    json.dump(result, file, indent=4)

print(f"Data has been converted and saved to {output_file}")
