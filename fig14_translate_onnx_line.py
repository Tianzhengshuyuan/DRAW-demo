import csv
import json
import os
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="fig14")
parser.add_argument('--output', type=str, default="onnx_line")

# 解析命令行参数
args = parser.parse_args()
input_name = args.input
output_name = args.output

# 动态加载 JSON 数据和设置保存路径
input_file = os.path.join("result", f"{input_name}.csv")   # 动态设置 csv文件
output_file = os.path.join("json", f"{output_name}.json")   # 动态设置 json文件


# 用于存储结果的列表
result = []

# 读取CSV文件
with open(input_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    
    # 跳过表头（如果有）
    header = next(reader)
    header = next(reader)
    
    row_count = 0
    A55_sum = 0
    A76_sum = 0
    M1P_sum = 0
    
    # 处理每一行数据
    for row in reader:
        # 获取category信息
        category = row[0]
        
        if row_count > 6:
            break
        row_count +=1
        
        # 计算 A55, A76, M1P 等的值
        try:
            A55 = float(row[4]) / float(row[3])   
            A76 = float(row[10]) / float(row[9])  
            M1P = float(row[16]) / float(row[15]) 
        except ZeroDivisionError:
            A55 = A76 = M1P = None  # 如果遇到除0错误，则设置为None
        
        # 构建JSON条目
        entry = {
            "category": category,
            "A55": A55,
            "A76": A76,
            "M1P": M1P
        }
        
        A55_sum += A55
        A76_sum += A76
        M1P_sum += M1P
        
        # 将条目添加到结果中
        result.append(entry)
        
    A55_geomean = A55_sum / row_count
    A76_geomean = A76_sum / row_count
    M1P_geomean = M1P_sum / row_count
    entry = {
        "category": 'geomean',
        "A55": A55_geomean,
        "A76": A76_geomean,
        "M1P": M1P_geomean
    }
    result.append(entry)

# 将结果保存为JSON文件
with open(output_file, 'w') as json_file:
    json.dump(result, json_file, indent=4)

print(f"处理完成，数据已保存到 {output_file}")
