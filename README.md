## SELF-J

This repository contains the code for our paper [SELF-J: Selective Instruction Following with Alignment Self-Evaluation](). 



## Quick Start
We provide one of our trained judge models and the code of generating quality scores with the judge model for evaluation. 

#### 1. Setup
We base on `vLLM` for inference, so you have to refer [here](https://docs.vllm.ai/en/latest/getting_started/installation.html) to install vLLM if you don't have. 

#### 2. Inference
We provide the inference code in `judge.py`. To get the score, run:
```bash
python judge.py
```

#### 3. Model
We release our judge model, tuned for the instruction-following model of Vicuna-v1.5, at Huggingface, link: [Self-J-13B-Vicuna-v1.5](https://huggingface.co/oceanpty/Self-J). 


## Judge Model Tuning
Our training code for judge modeling is based on the project of [Alpaca-Lora](https://github.com/tloen/alpaca-lora), so you will have to first follow the original instructions to set up the environment. 


#### Training Data
We provide the example training data for tuning the judge model at Huggingface, where the evaluated model is Vicuna-v1.5 and the qulaity score is the combination of model's self-evaluation and cosine similarity. 

1. Data with reference answer: [data_w_ref](https://huggingface.co/datasets/oceanpty/self-j/blob/main/training_set.round1.cosine_reivew.w_ref.category%3D1-10.review_by_vicuna-13b-v1.5.base_model%3Dvicuna-13b-v1.5.num%3D30k.json). 

2. Data without reference answer: [data_wo_ref](https://huggingface.co/datasets/oceanpty/self-j/blob/main/training_set.round1.cosine_reivew.wo_ref.category%3D1-10.review_by_vicuna-13b-v1.5.base_model%3Dvicuna-13b-v1.5.num%3D30k.json). 



## Instruction Collection
### Statistics
We collect a large-scale of instructions to study alignment evaluation on generation tasks, such as coding, writing, etc. We manually filtered datasets from Hugging Face as of June 2023, particularly those in the NLP category. 
We post-processed the datasets to filter out low-quality instructions as much as possible. 
We retained all good-quality instructions. We removed instructions that were either too short or too long. We also used the original instructions without tokenization, paraphrasing, etc, to maintain the real distribution of the instructions. After sorting, we keep 37 datasets in total. We manually categorized the datasets into three main categories: common, coding, and academic. Common instructions mainly concern everyday matters, such as seeking advice and solving technical problems. All instructions involving coding such as code generation and debugging are classified under the coding category. Lastly, subject-specific instructions, such as science and medicine, are categorized as academic. 

![](./figures/instruction_statistics.png)

### Data Release
We have released the collection of instructions, and you can download the data from Huggingface at [instruction-5.7m](https://huggingface.co/datasets/oceanpty/Self-J/blob/main/large_scale_instruction_collection.num%3D5754412.jsonl). 


### Performance: AlpacaEval
By fine-tuning Llama-2-13b with random 87K of our instruction selections and
GPT-3.5 Turbo responses, we can match Llama-2-13b-Chat's performance on AlpacaEval.
| Models | V1 | v2 |
| :------------- | :----------------: | :--------: | 
|Vicuna 13B v1.5    |  - | 6.7 |
| Llama-2-13B-Chat    |  81.09 |     7.7    |    
| Ours-Llama-2-13B     |  79.13  |     7.33     |   






