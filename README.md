## SELF-J: Selective Instruction Following with Alignment Self-Evaluation

This repository contains the code for our paper [SELF-J: Selective Instruction Following with Alignment Self-Evaluation](). 



## Quick Start

### Requirements

Our training code for judge modeling is based on the project of Alpaca-Lora, so you will have to first follow the original instructions to set up the environment. 

After downloding Alpaca-Lora, and suppose the file name is Alpaca-Lora, copy and put our code under ./Alpaca-Lora.  

### Download the data

We provide the example training data for tuning the judge model, where the evaluated model is Vicuna-v1.5 and the qulaity score is the combination of model's self-evaluation and cosine similarity. Download the data with reference answer at [w_ref](https://huggingface.co/datasets/oceanpty/self-j/blob/main/training_set.round1.cosine_reivew.w_ref.category%3D1-10.review_by_vicuna-13b-v1.5.base_model%3Dvicuna-13b-v1.5.num%3D30k.json), and the data without references at [wo_ref](https://huggingface.co/datasets/oceanpty/self-j/blob/main/training_set.round1.cosine_reivew.wo_ref.category%3D1-10.review_by_vicuna-13b-v1.5.base_model%3Dvicuna-13b-v1.5.num%3D30k.json). 


