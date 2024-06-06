import os
import sys
from typing import List
import fire
import torch
import transformers
from datasets import load_dataset

from transformers import DataCollatorForSeq2Seq 
from torch.utils.data import Dataset
import numpy as np


from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter
from kd import Trainer

class CustomDataCollatorWithPadding(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch1, batch2 = zip(*features) 

        batch1 = super().__call__(list(batch1))
        batch2 = super().__call__(list(batch2))

        combined_batch = (batch1, batch2)

        return combined_batch


class PairedDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        if dataset1 is None or dataset2 is None:
            return

        self.dataset1 = dataset1
        self.dataset2 = dataset2
        ids1 = [item['id'] for item in dataset1]
        dataset2_map = {item['id']: item for item in dataset2}
        self.dataset2 = [dataset2_map[id] for id in ids1]
        assert all(id in dataset2_map for id in ids1), "Some IDs in dataset1 are not in dataset2"
        assert len(dataset1) == len(dataset2), "Datasets must be of equal length"

    def __len__(self):
        return len(self.dataset1)

    def __getitem__(self, idx):
        data1 = self.dataset1[idx]
        data2 = self.dataset2[idx]

        if data1['id'] != data2['id']:
            print(f"Mismatched IDs at index {idx}: data1 ID = {data1['id']}, data2 ID = {data2['id']}")
        assert data1['id'] == data2['id'], "IDs do not match"

        data1 = {
            'input_ids': data1['input_ids'],
            'attention_mask': data1['attention_mask'],
            'labels': data1['labels']
        }
        data2 = {
            'input_ids': data2['input_ids'],
            'attention_mask': data2['attention_mask'],
            'labels': data2['labels']
        }

        return data1, data2


def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path_T: str = "yahma/alpaca-cleaned",
    data_path_S: str = "yahma/alpaca-cleaned",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = False,  # if False, masks out inputs in loss
    add_eos_token: bool = True, ## the original is False
    group_by_length: bool = False,  # faster, but produces an odd training loss curve,
    eval_steps: int = 200,
    save_steps: int = 200,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    cache_dir: str = "~/.cache"
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"data_path_T: {data_path_T}\n"
            f"data_path_S: {data_path_S}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"lora_target_modules: {lora_target_modules}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"add_eos_token: {add_eos_token}\n"
            f"group_by_length: {group_by_length}\n"
            f"wandb_project: {wandb_project}\n"
            f"wandb_run_name: {wandb_run_name}\n"
            f"wandb_watch: {wandb_watch}\n"
            f"wandb_log_model: {wandb_log_model}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
            f"prompt template: {prompt_template_name}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(prompt_template_name)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left" # the original is "left"  # Allow batched inference

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably


        return  tokenized_full_prompt

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    
    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if data_path_T.endswith(".json") or data_path_T.endswith(".jsonl"):
        data_T = load_dataset("json", data_files=data_path_T, cache_dir=cache_dir)
    else:
        data_T = load_dataset(data_path_T, cache_dir=cache_dir)
    
    if data_path_S.endswith(".json") or data_path_S.endswith(".jsonl"):
        data_S = load_dataset("json", data_files=data_path_S, cache_dir=cache_dir)
    else:
        data_S = load_dataset(data_path_S, cache_dir=cache_dir)
    

    if val_set_size > 0:
        train_val1 = data_T["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_val2 = data_S["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)

        train_data1 = train_val1["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data1 = train_val1["test"].shuffle().map(generate_and_tokenize_prompt)

        train_data2 = train_val2["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data2 = train_val2["test"].shuffle().map(generate_and_tokenize_prompt)
    else:

        train_data1 = data_T["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data1 = None
        train_data2 = data_S["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data2 = None

    
    paired_train_dataset = PairedDataset(train_data1, train_data2)
    paired_val_dataset = PairedDataset(val_data1, val_data2)




    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = Trainer(
        model=model,
        train_dataset=paired_train_dataset,
        eval_dataset=paired_val_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=CustomDataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )


if __name__ == "__main__":
    fire.Fire(train)
