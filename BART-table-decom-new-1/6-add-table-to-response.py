import json
import pandas as pd

with open("./data/train_gpt_new_decom_1_1_14000.json", "r") as f:
    data = json.load(f)

with open("./data/train.json", "r") as f:
    data2 = json.load(f)

result = []
cnt = 0 

for d in data:
    cnt += 1
    table = data2[cnt - 1]["table"]
    markdown_table = pd.DataFrame(table).to_markdown(index=False)
    
    result.append({
        "idx": cnt,
        "text": d["response"][0], 
        "question": d["question"],
        "origin_table": data2[cnt-1]["table"],
        "table": d["table"],
        "answer": d["answer"]
    })

with open("./data/train_gpt_new_decom_fullinfo_1_14000.json", "w") as f:
    json.dump(result, f, indent=4)