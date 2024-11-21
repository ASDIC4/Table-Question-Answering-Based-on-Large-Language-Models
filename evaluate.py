from collections import Counter
from utils.eval import eval_ex_match, extract_answer
import random
import json
import numpy as np
from tqdm import tqdm
from fire import Fire
import re
from typing import Union, List, Tuple


def flatten(lst):
    flat_list = []
    for i in lst:
        if isinstance(i, list):
            flat_list.extend(flatten(i))
        else:
            flat_list.append(i)
    return flat_list

def load_results(checkpoints, elements_per_checkpoint):
    
#    print(f"Loading {checkpoints}...")
    # not a list or a tuple, make it a list
    if not isinstance(checkpoints, list) and not isinstance(checkpoints, tuple):
        # try to split by comma
        if "," in checkpoints:
            checkpoints = checkpoints.split(",")
            # remove the spaces
            checkpoints = [checkpoint.strip() for checkpoint in checkpoints]
        else:
            checkpoints = [checkpoints]
    
    all_results = []
    
    # read all checkpoints
    for checkpoint in checkpoints:
        # print(f"Loading {checkpoint}...")
        
        if checkpoint.endswith(".jsonl"):
            with open(checkpoint, "r") as f:
                results = [json.loads(line) for line in f.readlines()]
        else:
            with open(f"output/{checkpoint}/result.jsonl", "r") as f:
                results = [json.loads(line) for line in f.readlines()]
        
        # print(f"Loaded {len(results)} results.")
        
        
        # deduplicate the results by id
        results = {result["question_id"]: result for result in results}
        results = list(results.values())
        
        all_results.append(results)
    
    # make sure the checkpoints are same length, if not, cut the longer one
    min_len = min([len(results) for results in all_results])
    all_results = [results[:min_len] for results in all_results]
    
    # the results are now in the form of [[dict, dict, ...], [dict, dict, ...], ...]
    # we want to combine them into one list of dicts by aggregating the dict["text"] field
    combined_results = []
    for i, results in enumerate(all_results):
        if i == 0:
            # if this is the first checkpoint, just add the results
            combined_results = results
            # make the text field a list of list
            for result in combined_results:
                # random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                result["text"] = random.sample(result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else [result["text"]]
            
        else:
            # if this is not the first checkpoint, add the text field to the existing list
            for j, result in enumerate(results):
                # remember to random sample the text field if specified
                if isinstance(result["text"], str):
                    result["text"] = [result["text"]]
                temp = random.sample(result["text"], elements_per_checkpoint[i]) if elements_per_checkpoint else result["text"]
                
                # add by question id instead of index
                for k, combined_result in enumerate(combined_results):
                    if combined_result["question_id"] == result["question_id"]:
                        combined_results[k]["text"].append(temp)
                        break
    
    # now we have a list of dicts with the text field being a list of list
    return combined_results
                
def eval_wtq(checkpoints:Union[List, Tuple, str], elements_per_checkpoint:Union[None, int, List]=None, n_times: int = 1, sub_sample_question_ids: list = None):
    # print("🚀 Starting Evaluation...")
    results = load_results(checkpoints, elements_per_checkpoint)
    print("len of results",len(results))
    # load_results function would merge multiple running results into one list
    # print(f"😂 Great Start: {len(results)}")
    
#    print(f"len of results: {len(results)}")
    acc_list = []
    for i in tqdm(range(n_times), desc="Progress", unit="batch"):
        acc, total = 0, 0

        # results的len 是 172 questions（20 tables）
        for result in results:
            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                continue
            answer = ", ".join(result["answer"])
            if isinstance(result["text"], str):
                result["text"] = [result["text"]]                    
            
            # Flatten the list to make sure it is 1D
            result["text"] = flatten(result["text"])
            
            preds = [extract_answer(text) for text in result["text"]]
            preds = [pred for pred in preds if pred]
            if n_times > 1:
                np.random.shuffle(preds)
            if not preds:
                total += 1
                continue
            
            # preds 是针对每一个问题，重复几次的，不同的pred
            # print(f"PRED: {preds}")

            # Majority voting
            pred_count = Counter(preds)
            pred, _ = pred_count.most_common(1)[0]

            if eval_ex_match(answer, pred):
                acc += 1
            total += 1

        acc_list.append(acc / total * 100)
        # print(f"acc: {acc}   total: {total}")

    # # print(acc_list)
    # print("🏁 Evaluation Complete.")

    # print(f"len of acc_list: {len(acc_list)}")
    # ## len depends on the n_times, not the 
    # # Print results
    # print(f"📊 Statistical Summary of {n_times} Trials on {total} Examples from Combined Checkpoints")
    # print(f"Min Accuracy: {min(acc_list):.2f}% ({np.round(min(acc_list) / 100 * total)}/{total})")
    # print(f"Max Accuracy: {max(acc_list):.2f}% ({np.round(max(acc_list) / 100 * total)}/{total})")
    
    print(f"Mean Accuracy: {np.mean(acc_list):.2f}% ({np.round(np.mean(acc_list) / 100 * total)}/{total})")
    print(f"Standard Deviation: {np.std(acc_list):.2f}%")


def eval_wtq_random(checkpoints:Union[List, Tuple, str], elements_per_checkpoint:Union[None, int, List]=None, n_times: int = 1, sub_sample_question_ids: list = None):
    # print("🚀 Starting Evaluation...")
    results = load_results(checkpoints, elements_per_checkpoint)
    # print("len of results",len(results))
    # load_results function would merge multiple running results into one list
    # print(f"😂 Great Start: {len(results)}")
    
#    print(f"len of results: {len(results)}")
    acc_list = []
    for i in range(n_times):
    # for i in tqdm(range(n_times), desc="Progress", unit="batch"):
        acc, total = 0, 0

        # results的len 是 172 questions（20 tables）
        for result in results:
            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                continue
            answer = ", ".join(result["answer"])
            if isinstance(result["text"], str):
                result["text"] = [result["text"]]                    
            
            # Flatten the list to make sure it is 1D
            result["text"] = flatten(result["text"])
            
            preds = [extract_answer(text) for text in result["text"]]
            preds = [pred for pred in preds if pred]
            if n_times > 1:
                np.random.shuffle(preds)
            if not preds:
                total += 1
                continue
            
            # preds 是针对每一个问题，重复几次的，不同的pred
            # print(f"PRED: {preds}")

            # Majority voting
            pred_count = Counter(preds)
            pred, _ = pred_count.most_common(1)[0]

            if eval_ex_match(answer, pred):
                acc += 1
            total += 1

        acc_list.append(acc / total * 100)
        # print(f"acc: {acc}   total: {total}")

    # # print(acc_list)
    # print("🏁 Evaluation Complete.")

    # print(f"len of acc_list: {len(acc_list)}")
    # ## len depends on the n_times, not the 
    # # Print results
    # print(f"📊 Statistical Summary of {n_times} Trials on {total} Examples from Combined Checkpoints")
    # print(f"Min Accuracy: {min(acc_list):.2f}% ({np.round(min(acc_list) / 100 * total)}/{total})")
    # print(f"Max Accuracy: {max(acc_list):.2f}% ({np.round(max(acc_list) / 100 * total)}/{total})")
    
    # print(f"Mean Accuracy: {np.mean(acc_list):.2f}% ({np.round(np.mean(acc_list) / 100 * total)}/{total})")
    # print(f"Standard Deviation: {np.std(acc_list):.2f}%")
    return np.mean(acc_list)

def markdown_table_to_list(markdown_table):
    # 使用正则表达式匹配Markdown表格中的行
    rows = re.findall(r'\|(.+?)\|', markdown_table, re.MULTILINE)
    
    # 去除每行中的空格并分割单元格内容
    table_data = [row.strip().split('|') for row in rows]
    
    # 去除首尾空白和空单元格
    table_data = [[cell.strip() for cell in row if cell.strip()] for row in table_data]
    
    return table_data

def eval_wtq_add_save(checkpoints:Union[List, Tuple, str], elements_per_checkpoint:Union[None, int, List]=None, n_times: int = 1, sub_sample_question_ids: list = None, save_path = None):
    # print("🚀 Starting Evaluation...")
    results = load_results(checkpoints, elements_per_checkpoint)
    # load_results function would merge multiple running results into one list
    # print(f"😂 Great Start: {len(results)}")
    
    print(f"len of results: {len(results)}")
    acc_list = []
    for i in tqdm(range(n_times), desc="Progress", unit="batch"):
        acc, total = 0, 0

        eval_res = []
        # results的len 是 172 questions（20 tables）
        for result in results:
            # print("type",type(result["answer"]))
            # # tmp = str(results["answer"])

            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                print("continue")
                continue
            
            answer = ", ".join(result["answer"])
            
            if isinstance(result["text"], str):
                result["text"] = [result["text"]]                    
            
            # Flatten the list to make sure it is 1D
            result["text"] = flatten(result["text"])
            # print("result[text]", result["text"])
            
            preds = [extract_answer(text) for text in result["text"]]
            # print("preds", preds)
            preds = [pred for pred in preds if pred]
            if n_times > 1:
                np.random.shuffle(preds)
                
            if not preds:
                total += 1

                eval_res.append({
                    "idx": total,
                    "match": 0,
                    "pred": "",
                    "answer": answer,
                    "question": result["question"],
                    "table": result["table"]

                #   "table": markdown_table_to_list(result["table"])
                })
                # print("None")
                continue
            # print("preds",preds)
            
            # preds 是针对每一个问题，重复几次的，不同的pred
            # print(f"PRED: {preds}")

            # Majority voting
            pred_count = Counter(preds)
            pred, _ = pred_count.most_common(1)[0]

            # print("answer",answer)
            # print("pred",pred)
            fg = 0
            if eval_ex_match(answer, pred):
                acc += 1
                fg = 1
            total += 1

            eval_res.append({
                "idx": total,
                "match": fg,
                "pred": pred,
                "answer": answer,
                "question": result["question"],
                "table": result["table"]

             #   "table": markdown_table_to_list(result["table"])
            })

        acc_list.append(acc / total * 100)

        with open(save_path, "w") as f:
            json.dump(eval_res, f, indent=4)
        # print(f"acc: {acc}   total: {total}")
        
    # # print(acc_list)
    # print("🏁 Evaluation Complete.")

    # print(f"len of acc_list: {len(acc_list)}")
    # ## len depends on the n_times, not the 
    # # Print results
    # print(f"📊 Statistical Summary of {n_times} Trials on {total} Examples from Combined Checkpoints")
    # print(f"Min Accuracy: {min(acc_list):.2f}% ({np.round(min(acc_list) / 100 * total)}/{total})")
    # print(f"Max Accuracy: {max(acc_list):.2f}% ({np.round(max(acc_list) / 100 * total)}/{total})")
    
    print(f"Mean Accuracy: {np.mean(acc_list):.2f}% ({np.round(np.mean(acc_list) / 100 * total)}/{total})")
    print(f"Standard Deviation: {np.std(acc_list):.2f}%")


def eval_tabfact_add_save(checkpoints:Union[List, Tuple, str], elements_per_checkpoint:Union[None, int, List]=None, n_times: int = 1, sub_sample_question_ids: list = None, save_path = None):
    # print("🚀 Starting Evaluation...")
    results = load_results(checkpoints, elements_per_checkpoint)
    # load_results function would merge multiple running results into one list
    # print(f"😂 Great Start: {len(results)}")
    
    print(f"len of results: {len(results)}")
    acc_list = []
    for i in tqdm(range(n_times), desc="Progress", unit="batch"):
        acc, total = 0, 0

        eval_res = []
        # results的len 是 172 questions（20 tables）
        for result in results:
            print("type",type(result["answer"]))
            # tmp = str(results["answer"])

            if sub_sample_question_ids and result["question_id"] not in sub_sample_question_ids:
                print("continue")
                continue
            # print(result)
            # print("type",type(result["answer"]))
            # # tmp = str(results["answer"])
            
            answer = str(result["answer"])
            if answer == "1":
                answer = "yes"
            elif answer == "0":
                answer = "no"

            if isinstance(result["text"], str):
                result["text"] = [result["text"]]                    
            
            # Flatten the list to make sure it is 1D
            result["text"] = flatten(result["text"])
            # print("result[text]", result["text"])
            
            preds = [extract_answer(text) for text in result["text"]]
            # print("preds", preds)
            preds = [pred for pred in preds if pred]
            if n_times > 1:
                np.random.shuffle(preds)
            if not preds:
                total += 1
                eval_res.append({
                    "idx": total,
                    "match": fg,
                    "pred": "",
                    "answer": answer,
                    "question": result["question"],
                    "table": result["table"]

                #   "table": markdown_table_to_list(result["table"])
                })
                print("None")
                continue
            # print("preds",preds)
            
            # preds 是针对每一个问题，重复几次的，不同的pred
            # print(f"PRED: {preds}")

            # Majority voting
            pred_count = Counter(preds)
            pred, _ = pred_count.most_common(1)[0]

            # print("answer",answer)
            # print("pred",pred)
            fg = 0
            if eval_ex_match(answer, pred):
                acc += 1
                fg = 1
            total += 1

            eval_res.append({
                "idx": total,
                "match": fg,
                "pred": pred,
                "answer": answer,
                "question": result["question"],
                "table": result["table"]

             #   "table": markdown_table_to_list(result["table"])
            })

        acc_list.append(acc / total * 100)

        with open(save_path, "w") as f:
            json.dump(eval_res, f, indent=4)
        # print(f"acc: {acc}   total: {total}")
        
    # # print(acc_list)
    # print("🏁 Evaluation Complete.")

    # print(f"len of acc_list: {len(acc_list)}")
    # ## len depends on the n_times, not the 
    # # Print results
    # print(f"📊 Statistical Summary of {n_times} Trials on {total} Examples from Combined Checkpoints")
    # print(f"Min Accuracy: {min(acc_list):.2f}% ({np.round(min(acc_list) / 100 * total)}/{total})")
    # print(f"Max Accuracy: {max(acc_list):.2f}% ({np.round(max(acc_list) / 100 * total)}/{total})")
    
    print(f"Mean Accuracy: {np.mean(acc_list):.2f}% ({np.round(np.mean(acc_list) / 100 * total)}/{total})")
    print(f"Standard Deviation: {np.std(acc_list):.2f}%")

# Example usage:
# eval_wtq(checkpoints="checkpoint1.jsonl")
# eval_wtq(checkpoints=["checkpoint1.jsonl", "checkpoint2.jsonl"], elements_per_checkpoint=[5, 5])

if __name__ == "__main__":
    Fire(eval_wtq)