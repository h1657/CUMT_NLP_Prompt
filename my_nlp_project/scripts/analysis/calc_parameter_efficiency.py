#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import os
import sys
sys.path.append('/root/autodl-tmp/OpenPrompt')

from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, PtuningTemplate, ManualTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, ProtoVerbalizer

def count_parameters(model, only_trainable=True):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())

def main():
    model_path = "/root/autodl-tmp/OpenPrompt/bert-base-uncased"
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)
    
    bert_total_params = count_parameters(plm, only_trainable=False)
    
    results = {}
    
    manual_template = ManualTemplate(tokenizer=tokenizer)
    manual_template.text = '{"placeholder": "text_a"} It was {"mask"}.'
    
    results['Manual_Template_Unfreeze'] = bert_total_params
    results['Manual_Template_Freeze'] = 0
    
    soft_template = SoftTemplate(
        model=plm, 
        tokenizer=tokenizer, 
        num_tokens=20,
        initialize_from_vocab=True
    )
    
    soft_template_params = count_parameters(soft_template, only_trainable=True)
    
    results['Soft_Template_Unfreeze'] = bert_total_params + soft_template_params
    results['Soft_Template_Freeze'] = soft_template_params
    
    soft_tokens = " ".join([f'{{"soft": "token_{i}"}}' for i in range(20)])
    ptuning_text = f'{{"placeholder": "text_a"}} {soft_tokens} {{"mask"}}.'
    
    ptuning_template = PtuningTemplate(
        model=plm,
        tokenizer=tokenizer,
        text=ptuning_text,
        prompt_encoder_type="lstm"
    )
    
    ptuning_params = count_parameters(ptuning_template, only_trainable=True)
    
    results['Ptuning_Template_Unfreeze'] = bert_total_params + ptuning_params
    results['Ptuning_Template_Freeze'] = ptuning_params
    
    manual_verbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=["negative", "positive"])
    manual_verb_params = count_parameters(manual_verbalizer, only_trainable=True)
    
    soft_verbalizer = SoftVerbalizer(
        tokenizer=tokenizer, 
        model=plm,
        classes=["negative", "positive"],
        label_words=["terrible", "great"]
    )
    soft_verb_params = count_parameters(soft_verbalizer, only_trainable=True)
    
    output_file = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/raw_data/parameter_statistics.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OpenPrompt 参数量统计结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"BERT-base总参数量: {bert_total_params:,}\n\n")
        
        f.write("详细参数统计:\n")
        for method, params in results.items():
            ratio = params / bert_total_params * 100
            f.write(f"{method}: {params:,} ({ratio:.2f}%)\n")
    
    return results

if __name__ == "__main__":
    results = main()
