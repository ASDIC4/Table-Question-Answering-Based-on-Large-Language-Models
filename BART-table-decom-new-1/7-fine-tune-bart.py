import torch
import json
from transformers import Trainer, TrainingArguments
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse



# Define your training dataset class
class MyDataset(Dataset):
    def __init__(self, data, label, tokenizer):
        self.data = data
        self.label = label
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Tokenize input sequence and target sequence using the tokenizer
        input = self.tokenizer.encode_plus(self.data[idx], add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        label = self.tokenizer.encode_plus(self.label[idx], add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')

        return {
            'input_ids': input['input_ids'].squeeze(),
            'attention_mask': input['attention_mask'].squeeze(),
            'labels': label['input_ids'].squeeze()
        }
        # return {
        #     'input_ids': inputs.input_ids.squeeze(),
        #     'attention_mask': inputs.attention_mask.squeeze(),
        #     'decoder_input_ids': target.input_ids.squeeze()[:-1],  # Remove the last token (eos_token_id)
        #     'decoder_attention_mask': target.attention_mask.squeeze()[:-1],  # Remove the last token's attention mask
        #     'labels': target.input_ids.squeeze()[1:],  # Shift target sequence to the right for teacher forcing
        # }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process start and end parameters.')
    parser.add_argument('--bar', type=int, help='start index', required=True)
    args = parser.parse_args()

    # Load and preprocess your training data
    with open('./data/train_gpt_new_decom_fullinfo_1_14000.json', "r") as f:
        input_data = json.load(f)
    
    pre_prompt_origin = """ 
    You are an advanced AI capable of analyzing and understanding information within tables.
    Here is a table, and a question related to table needed to be answered.
    Table header:
    [HEADER]

    Question:
    [QUESTION]
    
    I need you to tell me the columns that is needed to answer the question.
    Sometimes the question is about a whole row, which means the question still needs all the columns.
    Sometimes 'Name' column(if had) is implicitly needed.

    Ensure the final answer format is only "Final Answer: columns: ['column1', 'column2', ...]" 
    """

    # # 
    # Answer:
    # [ANSWER]

    train_data = []
    train_label = []
    
    bar = args.bar
    cnt = 0
    for d in input_data:
        cnt += 1
        if cnt > bar:
            continue
        # print("cnt",cnt)
        # print(d["origin_table"][0])
        # # print(",".join(d["origin_table"][0])  )
        prompt = pre_prompt_origin.replace("[HEADER]", ",".join(d["origin_table"][0]) )\
                .replace("[QUESTION]", d["question"])\
                .strip()
        # .replace("[ANSWER]", d["answer"])\
                
        train_data.append(prompt)
        train_label.append(d["text"])
    
    # Load pre-trained BERT model and tokenizer
    model_name = './bart-base'
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)

    train_dataset = MyDataset(train_data, train_label, tokenizer)
    # dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: list(zip(*x)))  # Use collate_fn for custom batch processing

    # Fine-tune the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print("device", device)
    import torch
    print(torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    training_args = TrainingArguments(
        output_dir='./results',
        # num_train_epochs=50,
        num_train_epochs=30,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=500,
    )
    def compute_loss(model, inputs):
        labels = inputs['labels']
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=labels
        )
        return outputs.loss
    
    print("Start training")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.compute_loss = compute_loss  # 设置compute_loss函数
    print("Before trainer.train() ")
    trainer.train()
    print("End trainer.train()")
    # model.save_pretrained('./new_decom_1_bart')
    # model.save_pretrained('./new_decom_1_bart_14000_epoch50')
    model.save_pretrained('./new_decom_1_bart_14000_epoch50_no-use-answer')
    
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch + 1}/{num_epochs}")
    #     model.train()
    #     total_loss = 0
    #     with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
    #         for batch in dataloader:
    #             inputs = batch['input_ids'].to(device)
    #             labels = batch['labels'].to(device)

    #             # input = torch.tensor(input).to(device)
    #             # label = torch.tensor(label).to(device)  # Convert responses to tensor and move to device

    #             optimizer.zero_grad()
    #             output = model(input_ids=inputs, labels=labels)
                
    #             loss = output.loss
    #             total_loss += loss.item()
    #             loss.backward()
    #             optimizer.step()

    #             pbar.update(1)  # Update progress bar

    #     average_loss = total_loss / len(dataloader)
    #     print(f"Average Loss: {average_loss}")

    # Save the fine-tuned model
