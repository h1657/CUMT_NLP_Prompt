#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
from collections import defaultdict
import sys

sys.path.append('/root/autodl-tmp/OpenPrompt')

from openprompt import PromptForClassification, PromptDataLoader
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, SoftTemplate, PtuningTemplate
from openprompt.prompts import ManualVerbalizer, SoftVerbalizer, ProtoVerbalizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import yaml

class ModelInference:
    def __init__(self, base_path="/root/autodl-tmp/OpenPrompt/my_nlp_project"):
        self.base_path = base_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_data = None
        
    def load_test_data(self):
        try:
            test_file = os.path.join(self.base_path, "data", "test.tsv")
            
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.test_data = []
            
            for i, line in enumerate(lines[:200]):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    text = parts[0]
                    label = int(parts[1])
                    
                    self.test_data.append(InputExample(
                        guid=str(i),
                        text_a=text,
                        label=label
                    ))
            
            return True
            
        except Exception as e:
            return False
    
    def load_finetuning_model(self):
        try:
            model_path = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/finetuning_baseline/checkpoint-1299"
            
            if not os.path.exists(model_path):
                return None, None
            
            local_bert_path = "/root/autodl-tmp/OpenPrompt/bert-base-uncased"
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(local_bert_path)
            except Exception as e:
                return None, None
            
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.to(self.device)
            model.eval()
            
            return model, tokenizer
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None
    
    def load_prompt_model(self, config_path, checkpoint_base_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            plm, tokenizer, model_config, WrapperClass = load_plm("bert", 
                "/root/autodl-tmp/OpenPrompt/bert-base-uncased")
            
            template, verbalizer = self.create_template_verbalizer(config, plm, tokenizer, WrapperClass)
            
            prompt_model = PromptForClassification(
                plm=plm,
                template=template,
                verbalizer=verbalizer,
                freeze_plm=config.get('plm', {}).get('optimize', {}).get('freeze_para', False)
            )
            
            checkpoint_path = None
            possible_paths = [
                f"{checkpoint_base_path}/seed-42/checkpoints/best.ckpt",
                f"{checkpoint_base_path}/seed-123/checkpoints/best.ckpt", 
                f"{checkpoint_base_path}/seed-456/checkpoints/best.ckpt",
                f"{checkpoint_base_path}/seed-42/checkpoints/last.ckpt",
                f"{checkpoint_base_path}/seed-123/checkpoints/last.ckpt", 
                f"{checkpoint_base_path}/seed-456/checkpoints/last.ckpt",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif isinstance(checkpoint, dict) and any(key.startswith('prompt_model.') for key in checkpoint.keys()):
                    state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                try:
                    missing_keys, unexpected_keys = prompt_model.load_state_dict(state_dict, strict=False)
                except Exception as e:
                    pass
            
            prompt_model.to(self.device)
            prompt_model.eval()
            
            return prompt_model, tokenizer, WrapperClass
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def create_template_verbalizer(self, config, plm, tokenizer, WrapperClass):
        template_type = config.get('template')
        
        if template_type == 'manual_template':
            template_file = config['manual_template']['file_path']
            if not os.path.exists(template_file):
                raise FileNotFoundError(f"模板文件不存在: {template_file}")
            template_text = open(template_file).read().strip()
            template = ManualTemplate(
                text=template_text,
                tokenizer=tokenizer
            )
        elif template_type == 'soft_template':
            soft_template_config = config.get('soft_template', {})
            
            if 'file_path' in soft_template_config and os.path.exists(soft_template_config['file_path']):
                with open(soft_template_config['file_path'], 'r') as f:
                    template_text = f.read().strip()
            else:
                template_text = '{"placeholder":"text_a"} {"soft"} {"mask"}'
            
            template = SoftTemplate(
                model=plm,
                tokenizer=tokenizer,
                text=template_text,
                num_tokens=soft_template_config.get('soft_token_num', 20),
                initialize_from_vocab=soft_template_config.get('initialize_from_vocab', True)
            )
        elif template_type == 'ptuning_template':
            ptuning_config = config.get('ptuning_template', {})
            if 'file_path' in ptuning_config and os.path.exists(ptuning_config['file_path']):
                with open(ptuning_config['file_path'], 'r') as f:
                    template_text = f.read().strip()
            elif 'text' in ptuning_config:
                template_text = ptuning_config['text']
            else:
                template_text = '{"soft": "It was"} {"mask"} {"soft": "."}'
            
            template = PtuningTemplate(
                model=plm,
                tokenizer=tokenizer,
                text=template_text,
                prompt_encoder_type=ptuning_config.get('encoder_type', 'lstm')
            )
        else:
            raise ValueError(f"未知的模板类型: {template_type}")
        
        verbalizer_type = config.get('verbalizer')
        
        if verbalizer_type == 'manual_verbalizer':
            verbalizer_file = config['manual_verbalizer']['file_path']
            if not os.path.exists(verbalizer_file):
                raise FileNotFoundError(f"Verbalizer文件不存在: {verbalizer_file}")
            with open(verbalizer_file) as f:
                lines = f.readlines()
            
            if len(lines) >= 2:
                label_words = [['terrible'], ['great']]
            else:
                label_words = [[line.strip()] for line in lines if line.strip()]
            
            verbalizer = ManualVerbalizer(
                classes=['0', '1'],
                label_words=label_words,
                tokenizer=tokenizer
            )
        elif verbalizer_type == 'soft_verbalizer':
            verbalizer = SoftVerbalizer(
                classes=['0', '1'],
                model=plm,
                tokenizer=tokenizer
            )
        elif verbalizer_type in ['prototypical_verbalizer', 'proto_verbalizer']:
            verbalizer = ProtoVerbalizer(
                classes=['0', '1'],
                model=plm,
                tokenizer=tokenizer
            )
        else:
            raise ValueError(f"未知的verbalizer类型: {verbalizer_type}")
        
        return template, verbalizer
    
    def predict_finetuning(self, model, tokenizer, examples):
        predictions = []
        confidences = []
        
        with torch.no_grad():
            for example in tqdm(examples, desc="Fine-tuning预测"):
                inputs = tokenizer(
                    example.text_a,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                
                pred = torch.argmax(logits, dim=-1).item()
                conf = torch.max(probs, dim=-1)[0].item()
                
                predictions.append(pred)
                confidences.append(conf)
        
        return predictions, confidences
    
    def predict_prompt(self, model, template, tokenizer, WrapperClass, examples):
        predictions = []
        confidences = []
        
        try:
            dataloader = PromptDataLoader(
                dataset=examples,
                template=template,
                tokenizer=tokenizer,
                tokenizer_wrapper_class=WrapperClass,
                max_seq_length=256,
                batch_size=1,
                shuffle=False,
                teacher_forcing=False,
                predict_eos_token=False,
                truncate_method="head"
            )
            
            with torch.no_grad():
                for step, inputs in enumerate(tqdm(dataloader, desc="提示模型预测")):
                    try:
                        if torch.cuda.is_available():
                            inputs = inputs.to(self.device)
                        
                        logits = model(inputs)
                        probs = torch.softmax(logits, dim=-1)
                        
                        pred = torch.argmax(logits, dim=-1).item()
                        conf = torch.max(probs, dim=-1)[0].item()
                        
                        predictions.append(pred)
                        confidences.append(conf)
                    except Exception as e:
                        predictions.append(0)
                        confidences.append(0.5)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            predictions = [0] * len(examples)
            confidences = [0.5] * len(examples)
        
        return predictions, confidences
    
    def run_inference(self):
        if not self.load_test_data():
            return
        
        results = {}
        
        predictions_dir = os.path.join(self.base_path, "outputs", "predictions")
        os.makedirs(predictions_dir, exist_ok=True)
        
        model_configs = {
            'finetuning': {
                'type': 'finetuning',
                'path': None
            },
            'manual_template': {
                'type': 'prompt',
                'config': f"{self.base_path}/configs/sst2_manual_prompt.yaml",
                'checkpoint': f"{self.base_path}/logs/sst2_bert_manual_template_manual_verbalizer_16shot"
            },
            'soft_template': {
                'type': 'prompt', 
                'config': f"{self.base_path}/configs/sst2_soft_prompt.yaml",
                'checkpoint': f"{self.base_path}/logs/sst2_bert_soft_template_manual_verbalizer_16shot"
            },
            'ptuning': {
                'type': 'prompt',
                'config': f"{self.base_path}/configs/sst2_ptuning.yaml", 
                'checkpoint': f"{self.base_path}/logs/sst2_bert_ptuning_template_manual_verbalizer_16shot"
            },
        }
        
        for model_name, config in model_configs.items():
            if config['type'] == 'finetuning':
                model, tokenizer = self.load_finetuning_model()
                if model is not None:
                    preds, confs = self.predict_finetuning(model, tokenizer, self.test_data)
                    results[model_name] = {
                        'predictions': preds,
                        'confidences': confs
                    }
                    del model, tokenizer
                    torch.cuda.empty_cache()
            else:
                if not os.path.exists(config['config']):
                    continue
                    
                result = self.load_prompt_model(config['config'], config['checkpoint'])
                if result[0] is not None:
                    model, tokenizer, WrapperClass = result
                    with open(config['config'], 'r') as f:
                        config_dict = yaml.safe_load(f)
                    
                    plm, _, _, _ = load_plm("bert", "/root/autodl-tmp/OpenPrompt/bert-base-uncased")
                    template, _ = self.create_template_verbalizer(config_dict, plm, tokenizer, WrapperClass)
                    
                    preds, confs = self.predict_prompt(model, template, tokenizer, WrapperClass, self.test_data)
                    results[model_name] = {
                        'predictions': preds,
                        'confidences': confs
                    }
                    del model, tokenizer, template, plm
                    torch.cuda.empty_cache()
        
        if results:
            self.save_predictions(results)
        
        return results
    
    def save_predictions(self, results):
        output_data = {
            'test_samples': [
                {
                    'id': example.guid,
                    'text': example.text_a,
                    'true_label': example.label
                }
                for example in self.test_data
            ],
            'predictions': results
        }
        
        output_path = f"{self.base_path}/outputs/predictions/all_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

def main():
    inference = ModelInference()
    results = inference.run_inference()

if __name__ == "__main__":
    main()