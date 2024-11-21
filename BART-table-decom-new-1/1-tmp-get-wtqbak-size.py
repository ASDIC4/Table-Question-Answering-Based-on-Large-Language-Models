import json

with open("/home/zjc/tablellm/data/wtq.json.bak","r") as f:
    data = json.load(f)

print(len(data))