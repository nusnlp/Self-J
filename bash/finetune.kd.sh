wandb disabled

GPU=1

source /home/yehai/miniconda3/bin/activate finetune-lora

prompt_template_name=wizardlm
root_path=/home/yehai/instruction-data-creation/instruction/build_hard_eval_set/hard_samples/adversarial/calibrate/exp_ood/

#for name in  WizardLM-13B-V1.2  vicuna-13b-v1.5  llama-2-13b-chat  llama-2-70b-chat  ; do

#for name in  WizardLM-13B-V1.2  vicuna-13b-v1.5  llama-2-13b-chat  llama-2-70b-chat   ; do

#for name in  WizardLM-13B-V1.2  vicuna-13b-v1.5     ; do
for name in  initial_model.num=80k  llama-2-13b-chat  llama-2-70b-chat  ; do

data_name_T=training_set.round1.cosine_reivew.w_ref.category=1-10.review_by_${name}.base_model=${name}.num=30k.json
data_name_S=training_set.round1.cosine_reivew.wo_ref.category=1-10.review_by_${name}.base_model=${name}.num=30k.json

data_path_T=$root_path/data/training_set_discriminator/$data_name_T
data_path_S=$root_path/data/training_set_discriminator/$data_name_S

base_model=/home/llm2/models/llama2/llama-2-13b-hf/
output_dir=$root_path/lora-weight/discriminator.distillation.T=round1.cosine_reivew.w_ref.S=round1.cosine_reivew.wo_ref.review_by_${name}.template=${prompt_template_name}.model=llama-2-13b.batch=128.lr=3e-4.epoch=2.alpha=0.3.alpha_for_kd=2/



root_alpaca_lora=/home/yehai/instruction-data-creation/finetune-lora/alpaca-lora/
cd $root_alpaca_lora
## The used hyper-parameters to finetune 13b
CUDA_VISIBLE_DEVICES=$GPU python finetune.kd.py \
    --base_model $base_model \
    --data_path_T $data_path_T \
    --data_path_S $data_path_S \
    --output_dir $output_dir \
    --prompt_template_name $prompt_template_name \
    --micro_batch_size 16 \
    --num_epochs 2 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,k_proj,v_proj,o_proj]' \
    --eval_steps 200 \
    --save_steps 200 \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --cache_dir '/localhome/yehai/cache'

done;