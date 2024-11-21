import os
import csv
import json

# 定义转换函数
def convert_to_json(input_file, output_file):
    data_list = []
    
    # 读取tsv文件并处理每一行数据
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')

        cnt = 0
    
        for row in reader:
            # 提取文件路径和问题答案
            file_path = os.path.join('./data', row['context'])
            
            cnt += 1
            
            # 读取对应文件中的内容，并以元组形式存储
            with open(file_path, 'r', encoding='utf-8') as f:
                csv_reader = csv.reader(f)
            
                table = []
                for csv_row in csv_reader:
                    # print(csv_row)
                    table.append(csv_row)

                data_list.append({
                    'idx': cnt,
                    'id': row['id'],
                    'question': row['utterance'],
                    'table': table,
                    'answer': row['targetValue'],
                    'file_path': file_path
                })
    
    # 将列表转换为JSON格式并保存到文件中
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

# 调用转换函数
convert_to_json('./data/training.tsv', './data/train.json')
