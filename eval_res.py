import os

from evaluate import eval_wtq, eval_wtq_add_save

checkpoints = [
    "",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_1/result-full.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_2/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_3/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_4/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_5/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_1/result-full.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_2/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_3/result-full.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_4/result.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_5/result.jsonl"
]
save_paths = [
    "",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_1/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_2/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_3/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_4/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/dp_decombart_5/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_1/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_2/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_3/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_4/eval_res.jsonl",
    "/home/zjc/zjc-grad/results/wtq_4k+_run/decompose/agent_decombart_5/eval_res.jsonl"
] 

name = [
    "",
    "textual_1",
    "textual_2",
    "textual_3",
    "textual_4",
    "textual_5",
    "symbolic_1",
    "symbolic_2",
    "symbolic_3",
    "symbolic_4",
    "symbolic_5"
]

# 单个的评价
for i in range(len(checkpoints)):
    if i == 0: 
        continue
    print(name[i])
    eval_wtq_add_save(checkpoints=checkpoints[i], n_times = 1, save_path=save_paths[i])
    print()

print("综合")
print()

print("1-5 mix: textual * 5 ")
eval_wtq(checkpoints=checkpoints[1:6], n_times=1)
print("6-10 mix: symbolic * 5 ")
eval_wtq(checkpoints=checkpoints[6:11], n_times=1)
print("1-10 mix: textual * 5 + symbolic * 5")
eval_wtq(checkpoints=checkpoints[1:11], n_times=1)
import os

# redirect to the parent directory
# os.chdir("..")

from evaluate import eval_wtq, eval_wtq_add_save

checkpoints = [
    "",
    "/home/zjc/zjc-grad/results/wtq-cot-all/result_single_1.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/result_single_2.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/result_single_3.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/result_single_4.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/result_single_5.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/result_sc1.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/result_sc2.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/result_sc3.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/result_sc4.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/result_sc5.jsonl"
]
save_paths = [
    "",
    "/home/zjc/zjc-grad/results/wtq-cot-all/eval_res_1.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/eval_res_2.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/eval_res_3.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/eval_res_4.jsonl",
    "/home/zjc/zjc-grad/results/wtq-cot-all/eval_res_5.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/eval_res_1.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/eval_res_2.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/eval_res_3.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/eval_res_4.jsonl",
    "/home/zjc/zjc-grad/results/wtq-agent-all/eval_res_5.jsonl"
] 

name = [
    "",
    "textual_1",
    "textual_2",
    "textual_3",
    "textual_4",
    "textual_5",
    "symbolic_1",
    "symbolic_2",
    "symbolic_3",
    "symbolic_4",
    "symbolic_5"
]

# 单个的评价
for i in range(len(checkpoints)):
    if i == 0: 
        continue
    print(name[i])
    eval_wtq_add_save(checkpoints=checkpoints[i], n_times = 1, save_path=save_paths[i])
    print()

print("综合")
print()

print("1-5 mix: textual * 5 ")
eval_wtq(checkpoints=checkpoints[1:6], n_times=1)
print("6-10 mix: symbolic * 5 ")
eval_wtq(checkpoints=checkpoints[6:11], n_times=1)
print("1-10 mix: textual * 5 + symbolic * 5")
eval_wtq(checkpoints=checkpoints[1:11], n_times=1)
