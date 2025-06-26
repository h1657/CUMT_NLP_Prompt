#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import yaml
from pathlib import Path
from transformers import BertTokenizer, BertModel
import sys
import os

sys.path.append('/root/autodl-tmp/OpenPrompt')
from openprompt.plms import load_plm
from openprompt.prompts import SoftTemplate, PtuningTemplate
from openprompt.config import get_user_config

class SoftPromptAnalyzer:
    def __init__(self, project_root='/root/autodl-tmp/OpenPrompt/my_nlp_project'):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'outputs'
        self.model_path = '/root/autodl-tmp/OpenPrompt/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        
    def load_model_and_prompt(self, experiment_name, prompt_type='soft_template'):
        if 'ptuning' in experiment_name:
            config_path = "/root/autodl-tmp/OpenPrompt/my_nlp_project/configs/sst2_ptuning_freeze.yaml"
        else:
            config_path = "/root/autodl-tmp/OpenPrompt/my_nlp_project/configs/sst2_soft_template_freeze.yaml"
            
        log_dir = self.project_root / f'logs/{experiment_name}_16shot/seed-456'
        
        checkpoint_patterns = [
            log_dir / 'checkpoints/best.ckpt',
            log_dir / 'checkpoints/last.ckpt', 
            log_dir / 'pytorch_model.bin',
            log_dir / 'best_model.pt'
        ]
        
        checkpoint_path = None
        for pattern in checkpoint_patterns:
            if pattern.exists():
                checkpoint_path = pattern
                break
                
        if checkpoint_path is None:
            all_files = list(log_dir.rglob('*.ckpt')) + list(log_dir.rglob('*.pt')) + list(log_dir.rglob('*.bin'))
            if all_files:
                checkpoint_path = all_files[0]
            else:
                return None, None
                
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {'soft_token_num': 20, 'init_from_vocab': True}
            
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", self.model_path)
        
        if 'ptuning' in experiment_name:
            template = PtuningTemplate(
                model=plm,
                tokenizer=tokenizer,
                prompt_encoder_type="lstm",
                text='{"placeholder": "text_a"} {"mask"}',
            )
        else:
            template = SoftTemplate(
                model=plm,
                tokenizer=tokenizer,
                num_tokens=config.get('soft_token_num', 20),
                initialize_from_vocab=config.get('init_from_vocab', True),
                text='{"soft": "soft_0", "duplicate": 19} {"placeholder": "text_a"} {"mask"}',
            )
            
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
        except Exception as e:
            return None, None
            
        soft_prompt_params = {}
        if 'ptuning' in experiment_name:
            for name, param in state_dict.items():
                if name.endswith('soft_embedding.weight') or name.endswith('new_embedding.weight'):
                    if hasattr(param, 'shape') and param.dim() == 2:
                        soft_prompt_params[name] = param
        else:
            for name, param in state_dict.items():
                if 'soft_embeds' in name or 'soft_token' in name:
                    soft_prompt_params[name] = param
                    
        if not soft_prompt_params:
            for name, param in state_dict.items():
                if 'template' in name.lower() and hasattr(param, 'shape') and param.dim() >= 2:
                    soft_prompt_params[name] = param
                    
        return soft_prompt_params, template
        
    def extract_embeddings(self, experiment_name, save_path=None):
        soft_params, template = self.load_model_and_prompt(experiment_name)
        
        if soft_params is None or len(soft_params) == 0:
            return None
            
        embeddings = {}
        if 'ptuning' in experiment_name:
            p_tuning_embeddings = []
            sorted_params = sorted(soft_params.items(), key=lambda x: x[0])
            for name, param in sorted_params:
                p_tuning_embeddings.append(param.detach().cpu())
            
            if p_tuning_embeddings:
                combined_embeddings = torch.cat(p_tuning_embeddings, dim=0)
                embeddings['ptuning_combined_embeddings'] = combined_embeddings.numpy()
            else:
                return None
        else:
            for name, param in soft_params.items():
                embeddings[name] = param.detach().cpu().numpy()
            
        if save_path is None:
            save_path = self.results_dir / 'embeddings' / f'{experiment_name}_embeddings.pkl'
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        return embeddings

if __name__ == "__main__":
    analyzer = SoftPromptAnalyzer()
    
    soft_embeddings = analyzer.extract_embeddings('sst2_bert_soft_template_freeze')
    
    ptuning_embeddings = analyzer.extract_embeddings('sst2_bert_ptuning_freeze')