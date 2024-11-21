import json

with open("./data/train_gpt_new_decom_1_1_1000.json", "r") as f:
    data1 = json.load(f)

with open("./data/train_gpt_new_decom_1_1001_2000.json", "r") as f:
    data2 = json.load(f)

with open("./data/train_gpt_new_decom_1_2001_3000.json", "r") as f:
    data3 = json.load(f)

with open("./data/train_gpt_new_decom_1_3001_3500.json", "r") as f:
    data4 = json.load(f)

with open("./data/train_gpt_new_decom_1_3501_4000.json", "r") as f:
    data5 = json.load(f)

with open("./data/train_gpt_new_decom_1_4001_5000.json", "r") as f:
    data6 = json.load(f)

with open("./data/train_gpt_new_decom_1_5001_6000.json", "r") as f:
    data7 = json.load(f)

with open("./data/train_gpt_new_decom_1_6001_7000.json", "r") as f:
    data8 = json.load(f)

with open("./data/train_gpt_new_decom_1_7001_8000.json", "r") as f:
    data9 = json.load(f)

with open("./data/train_gpt_new_decom_1_8001_9000.json", "r") as f:
    data10 = json.load(f)

with open("./data/train_gpt_new_decom_1_9001_10000.json", "r") as f:
    data11 = json.load(f)

with open("./data/train_gpt_new_decom_1_10001_11000.json", "r") as f:
    data12 = json.load(f)

with open("./data/train_gpt_new_decom_1_11001_12000.json", "r") as f:
    data13 = json.load(f)

with open("./data/train_gpt_new_decom_1_12001_13000.json", "r") as f:
    data14 = json.load(f)

with open("./data/train_gpt_new_decom_1_13001_14000.json", "r") as f:
    data15 = json.load(f)

from itertools import chain

data = list(chain(data1, data2, data3, data4, data5,
                  data6, data7, data8, data9, data10,
                  data6, data7, data8, data9, data15
                  ))

with open("./data/train_gpt_new_decom_1_1_14000.json", "w") as f:
    json.dump(data, f, indent=4)