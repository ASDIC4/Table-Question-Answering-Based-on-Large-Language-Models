import torch
import json
from tqdm import tqdm
import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration

def json_to_markdown(table):
    header = table["header"]
    rows = table["rows"]

    # Convert header to Markdown format
    header_md = "| " + " | ".join(header) + " |"
    separator = "| " + " | ".join([":" + "-" * (len(col) - 2) + ":" for col in header]) + " |"

    # Convert rows to Markdown format
    rows_md = []
    for row in rows:
        row_md = "| " + " | ".join(row) + " |"
        rows_md.append(row_md)

    # Combine header, separator, and rows into Markdown table
    markdown_table = "\n".join([header_md, separator] + rows_md)

    return markdown_table

def predict(input_data_path, model, tokenizer, output_data_path, questions_num):

    # pre_prompt_give_column_name = """ 
    # You are an advanced AI capable of analyzing and understanding information within tables. Read the column name of this table below.
    # [TABLE]
    # There is a question:
    # [QUESTION]
    # Please Give me some help. Think which columns or rows holds the information with higher probability. If the question has to answer use the whole table information to answer, just give me all the name of row and column.
    # Ensure the final answer format is only "Final Answer: row: ['row1', 'row2', ...], column: ['column1', 'column2', ...]" form, no other form. And ensure the final answer is a number or entity names, as short as possible, without any explanation.
    # """
    pre_prompt_origin = """ 
    You are an advanced AI capable of analyzing and understanding information within tables.
    Here is a table, and a question related to table needed to be answered.
    Table header:
    [HEADER]

    Question:
    [QUESTION]
    
    I need you to tell me the columns that is needed to answer the question.
    Sometimes the question is about a whole row, which means the question still needs all the columns.

    Ensure the final answer format is only "Final Answer: columns: ['column1', 'column2', ...]" 
    """

    # results = []

    with open(input_data_path, "r") as f:
        data = json.load(f)

   

    cnt = 0
    with tqdm(total=questions_num, desc="Processing") as pbar:
        for table_idx, d in enumerate(data):
            
            table = d["table"]
            markdown_table = json_to_markdown(table)

            response = []

            for idx in range(len(d["questions"])):
                cnt += 1
                
                question = d['questions'][idx]
                # answer = d['answers'][idx][0]

                question = d["questions"][idx]

                #                 .replace("[ANSWER]", answer) \

                prompt = pre_prompt_origin.replace("[HEADER]", ",".join(d["table"]["header"]))\
                .replace("[QUESTION]", question)\
                .strip()
                encoded_prompt = tokenizer.encode_plus(prompt, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

                # prompt = pre_prompt_origin.replace("[TABLE]", markdown_table)\
                #     .replace("[QUESTION]", question)\
                #     .strip()
                
                input_ids = encoded_prompt['input_ids'].to(device)
                attention_mask = encoded_prompt['attention_mask'].to(device)
            
                output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512, num_beams=5, early_stopping=True)
                decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            
                response.append(decoded_output)
                pbar.update(1) 

            d["response"] = response
    
    with open(output_data_path, "w") as f:
        json.dump(data, f, indent=4)
    

# def generate_predictions(data, tokenizer, model_path, predictions_path):
#     model = BartForConditionalGeneration.from_pretrained(model_path)

#     # 遍历数据，逐个样本进行预测
#     predictions = []
#     for text in tqdm(data, desc="Generating predictions"):
#         # 对当前样本进行预测
#         input_ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512,
#                                      padding="max_length", return_tensors="pt")
#         output = model.generate(input_ids, max_new_tokens=512)

#         prediction = tokenizer.decode(output[0], skip_special_tokens=True)

#         # 将预测结果添加到预测列表中
#         predictions.append(prediction)

#     # 保存预测数据到文件中
#     with open(predictions_path, "w") as f:
#         for prediction in predictions:
#             f.write(prediction + "\n")

if __name__ == "__main__":
    # # Load test data
    # with open('./data/test_data.json', "r") as f:
    #     test_data = json.load(f)

    # wtq: table with multiple questions

    print('===========================================')
    print('Predict Step')

    # Load the fine-tuned model for testing
    model_name = './new_decom_1_bart_14000_epoch50_no-use-answer'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained('./bart-base')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    test_data_path = "../data/wtq.json.bak.4k+.18"
    output_data_path = "../data/wtq.json.bak.4k+.18.add-new-decom-1-bart-no-use-answer.json"
    questions_num = 153
    # Perform prediction
    predict(test_data_path, model, tokenizer, output_data_path, questions_num=153)

    print("Prediction completed and saved.")