import json
import argparse
import os
import pandas as pd


# 命令行参数解析
parser = argparse.ArgumentParser(description="Generate plots for specific model data")
parser.add_argument('--input', type=str, default="fig7a" , help="The origin csv file name.")
parser.add_argument('--output', type=str, default="CpuFp16Speedup" , help="The origin csv file name.")

# 解析命令行参数
args = parser.parse_args()

input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件

# 读取CSV数据
# 读取CSV数据并跳过首行
df = pd.read_csv(
    input_file,
    names=["category", "TFLite", "MNN"],  # 指定列名
    skiprows=1                                      # 跳过第一行
)

# 将数据转换为列表字典格式
json_data = df.to_dict(orient="records")

# 将数据写入JSON文件
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False)

print("转换完成！JSON 文件已保存为", output_file)
