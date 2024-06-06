import torch
import torch.nn.functional as F
import transformers
from torch import nn
import sys
from collections import OrderedDict
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("big_bird", "BigBirdForCausalLM"),
        ("bigbird_pegasus", "BigBirdPegasusForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cpmant", "CpmAntForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("electra", "ElectraForCausalLM"),
        ("ernie", "ErnieForCausalLM"),
        ("git", "GitForCausalLM"),
        ("gpt-sw3", "GPT2LMHeadModel"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("mega", "MegaForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("open-llama", "OpenLlamaForCausalLM"),
        ("openai-gpt", "OpenAIGPTLMHeadModel"),
        ("opt", "OPTForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qdqbert", "QDQBertLMHeadModel"),
        ("reformer", "ReformerModelWithLMHead"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text_2", "Speech2Text2ForCausalLM"),
        ("transfo-xl", "TransfoXLLMHeadModel"),
        ("trocr", "TrOCRForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
    ]
)


class Trainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        def cal_one_input(input):
            if self.label_smoother is not None and "labels" in input:
                labels = input.pop("labels")
            else:
                labels = None

            outputs = model(**input)
        
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(input.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

            # Shift so that tokens < n predict n
            logits = outputs['logits']
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, logits.size()[2])
            return loss, shift_logits, outputs
        
        loss1, logits1, outputs = cal_one_input(inputs[0])
        loss2, logits2, outputs = cal_one_input(inputs[1])
        

        mask1 = inputs[0]['labels'][..., 1:].contiguous().ne(-100) & inputs[0]['labels'][..., 1:].contiguous().ne(2)
        mask1 = mask1.squeeze().view(-1)

        mask2 = inputs[1]['labels'][..., 1:].contiguous().ne(-100) & inputs[1]['labels'][..., 1:].contiguous().ne(2)
        mask2 = mask2.squeeze().view(-1)
        


        prob_teacher = F.softmax(logits1.detach()[mask1], dim=1)
        prob_student = F.softmax(logits2[mask2], dim=1)
        kl_div_loss = F.kl_div(prob_student.log(), prob_teacher, reduction='batchmean')
        alpha=0.3
        alpha_for_kd = 2
        loss = loss1 + alpha * loss2 + alpha_for_kd * kl_div_loss


        return (loss, outputs) if return_outputs else loss


