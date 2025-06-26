#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def collect_all_results():
    first_round_results = [
        {
            'method': 'Fine-tuning',
            'template': 'None',
            'verbalizer': 'None', 
            'freeze_plm': False,
            'accuracy': 91.40,
            'num_examples': 'Full',
            'experiment_type': 'baseline',
            'seeds_detail': [91.40]
        },
        {
            'method': 'Manual Template',
            'template': 'manual_template',
            'verbalizer': 'manual_verbalizer',
            'freeze_plm': False,
            'accuracy': 65.87,
            'num_examples': 16,
            'experiment_type': 'original',
            'seeds_detail': [65.88, 65.93, 65.77]
        },
        {
            'method': 'Soft Template',
            'template': 'soft_template', 
            'verbalizer': 'manual_verbalizer',
            'freeze_plm': False,
            'accuracy': 65.84,
            'num_examples': 16,
            'experiment_type': 'original',
            'seeds_detail': [66.65, 66.04, 64.84]
        },
        {
            'method': 'P-tuning',
            'template': 'ptuning_template',
            'verbalizer': 'manual_verbalizer', 
            'freeze_plm': False,
            'accuracy': 61.08,
            'num_examples': 16,
            'experiment_type': 'original',
            'seeds_detail': [61.48, 60.93, 60.82]
        },
        {
            'method': 'Proto Verbalizer',
            'template': 'manual_template',
            'verbalizer': 'proto_verbalizer',
            'freeze_plm': False,
            'accuracy': 56.23,
            'num_examples': 16,
            'experiment_type': 'original',
            'seeds_detail': [56.23]
        },
        {
            'method': 'Soft Verbalizer',
            'template': 'manual_template',
            'verbalizer': 'soft_verbalizer',
            'freeze_plm': False,
            'accuracy': 52.73,
            'num_examples': 16,
            'experiment_type': 'original',
            'seeds_detail': [52.73]
        }
    ]
    
    freeze_results = [
        {
            'method': 'Soft Template (Freeze)',
            'template': 'soft_template',
            'verbalizer': 'manual_verbalizer',
            'freeze_plm': True,
            'accuracy': 68.74,
            'num_examples': 16,
            'experiment_type': 'freeze',
            'seeds_detail': [71.26, 66.21, 68.74],
            'validation_scores': [72.56, 70.03, 70.15],
            'log_source': 'sst2_bert_soft_template_freeze_16shot'
        },
        {
            'method': 'P-tuning (Freeze)',
            'template': 'ptuning_template', 
            'verbalizer': 'manual_verbalizer',
            'freeze_plm': True,
            'accuracy': 57.05,
            'num_examples': 16,
            'experiment_type': 'freeze',
            'seeds_detail': [58.19, 55.66, 57.31],
            'validation_scores': [60.16, 56.83, 58.44],
            'log_source': 'sst2_bert_ptuning_freeze_16shot'
        }
    ]
    
    all_results = first_round_results + freeze_results
    return all_results

def estimate_parameters():
    bert_base_params = 109483778
    soft_prompt_params = 20 * 768
    ptuning_lstm_params = 500000  
    
    param_estimates = {
        'Fine-tuning': bert_base_params,
        'Manual Template': bert_base_params,
        'Soft Template': bert_base_params + soft_prompt_params,
        'P-tuning': bert_base_params + ptuning_lstm_params,
        'Proto Verbalizer': bert_base_params,
        'Soft Verbalizer': bert_base_params,
        'Soft Template (Freeze)': soft_prompt_params,
        'P-tuning (Freeze)': ptuning_lstm_params + soft_prompt_params,
    }
    
    return param_estimates

def calculate_statistics(results):
    stats = {}
    
    for result in results:
        if 'seeds_detail' in result and len(result['seeds_detail']) > 1:
            seeds = result['seeds_detail']
            stats[result['method']] = {
                'mean': np.mean(seeds),
                'std': np.std(seeds, ddof=1),
                'min': np.min(seeds),
                'max': np.max(seeds),
                'seeds': seeds
            }
        else:
            stats[result['method']] = {
                'mean': result['accuracy'],
                'std': 0.0,
                'min': result['accuracy'],
                'max': result['accuracy'],
                'seeds': [result['accuracy']]
            }
    
    return stats

def main():
    results = collect_all_results()
    param_estimates = estimate_parameters()
    stats = calculate_statistics(results)
    
    complete_results = []
    for result in results:
        params = param_estimates.get(result['method'], 0)
        method_stats = stats.get(result['method'], {})
        complete_results.append({
            'method': result['method'],
            'accuracy_mean': result['accuracy'],
            'accuracy_std': method_stats.get('std', 0),
            'seeds_detail': result.get('seeds_detail', []),
            'parameters': params,
            'parameter_efficiency_ratio': param_estimates['Fine-tuning'] / params if params > 0 else 0,
            'freeze_plm': result['freeze_plm'],
            'experiment_type': result['experiment_type'],
            'log_source': result.get('log_source', None)
        })
    
    with open('/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/raw_data/complete_results.json', 'w', encoding='utf-8') as f:
        json.dump(complete_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()