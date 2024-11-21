import json 
import re

def filter_table(header, rows, extracted_list):
    if set(extracted_list).issubset(header):
        # 如果提取的列表是表格的子集，则仅选择与提取的列表匹配的列
        filtered_header = extracted_list
        filtered_rows = [[row[header.index(col)] for col in extracted_list] for row in rows]
        return filtered_header, filtered_rows, 1
    else:
        # 如果提取的列表不是表格的子集，则保留原有的全部表格
        return header, rows, 0
    
def operate(input_path, output_path):

    everything_normal = 0

    with open(input_path, "r")as f:
        data = json.load(f)

    cnt = 0

    res = []
    for d in data:
        title = d["title"]
        tmp_len = len(d["questions"])
        table = d["table"]
        header = d["table"]["header"]
        rows = d["table"]["rows"]
        
        for idx in range(tmp_len):
            question = d["questions"][idx]
            answer = d["answers"][idx][0]
            response = d["response"][idx]
            question_id = d["ids"][idx]

                        # 使用正则表达式匹配列表部分
            match = re.search(r"\[.*?\]", response)

            # 提取匹配到的列表部分
            extracted_list = match.group(0)
            # 将提取的字符串转换为 Python 列表
            extracted_list = eval(extracted_list)
            # print("Extracted list as Python list:", extracted_list)
            
            filtered_header, filtered_rows, fg = filter_table(header, rows, extracted_list)
            everything_normal += fg
            tmp_table = {}
            tmp_table["header"] = filtered_header
            tmp_table["rows"] = filtered_rows
            
            res.append({
                "question": question,
                "answer": answer,
                "fg": fg,
                "response": extracted_list,
                "question_id": question_id,
                "table": tmp_table,
                "transposed_table": d["transposed_table"],
                "title": title,
                "row_shuffled_table": d["row_shuffled_table"],
                "row_shuffled_transposed_table": d["row_shuffled_transposed_table"]
            })
            cnt += 1

    with open(output_path, "w")as f:
        json.dump(res, f)

    print("normal", everything_normal)

if __name__ == "__main__":
    input_path = "/home/zjc/tablellm/data/wtq.json.bak.4k+.18.add-new-decom-1-bart-no-use-answer.json"
    output_path = "/home/zjc/tablellm/data/wtq.json.bak.4k+.18.add-new-decom-1-bart-no-use-answer-adjust.json"
    operate(input_path, output_path)