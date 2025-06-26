#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
from collections import defaultdict, Counter
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CaseStudyAnalyzer:
    def __init__(self, predictions_file):
        with open(predictions_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.test_samples = self.data['test_samples']
        self.predictions = self.data['predictions']
        self.model_names = list(self.predictions.keys())
        
        self.label_map = {0: 'Negative', 1: 'Positive'}
        
    def analyze_consistency(self):
        n_samples = len(self.test_samples)
        
        all_preds = {}
        for model in self.model_names:
            all_preds[model] = self.predictions[model]['predictions']
        
        consistent_correct = []
        consistent_wrong = []
        inconsistent = []
        
        for i in range(n_samples):
            true_label = self.test_samples[i]['true_label']
            model_preds = [all_preds[model][i] for model in self.model_names]
            
            all_correct = all(pred == true_label for pred in model_preds)
            all_wrong = all(pred != true_label for pred in model_preds)
            
            case_info = {
                'sample_id': i,
                'text': self.test_samples[i]['text'],
                'true_label': true_label,
                'true_label_name': self.label_map[true_label],
                'predictions': {model: all_preds[model][i] for model in self.model_names},
                'confidences': {model: self.predictions[model]['confidences'][i] for model in self.model_names}
            }
            
            if all_correct:
                consistent_correct.append(case_info)
            elif all_wrong:
                consistent_wrong.append(case_info)
            else:
                inconsistent.append(case_info)
        
        return {
            'consistent_correct': consistent_correct,
            'consistent_wrong': consistent_wrong,
            'inconsistent': inconsistent
        }
    
    def analyze_model_performance(self):
        performance = {}
        
        for model in self.model_names:
            preds = self.predictions[model]['predictions']
            true_labels = [sample['true_label'] for sample in self.test_samples]
            
            accuracy = sum(p == t for p, t in zip(preds, true_labels)) / len(true_labels)
            
            report = classification_report(true_labels, preds, output_dict=True)
            
            performance[model] = {
                'accuracy': accuracy,
                'precision': report['macro avg']['precision'],
                'recall': report['macro avg']['recall'],
                'f1': report['macro avg']['f1-score']
            }
        
        return performance
    
    def find_interesting_cases(self, consistency_results):
        interesting_cases = {
            'only_finetuning_correct': [],
            'only_prompts_correct': [],
            'best_prompt_vs_worst': [],
            'high_confidence_wrong': []
        }
        
        for case in consistency_results['inconsistent']:
            preds = case['predictions']
            confs = case['confidences']
            true_label = case['true_label']
            
            if (preds.get('finetuning', -1) == true_label and 
                all(preds[m] != true_label for m in self.model_names if m != 'finetuning')):
                interesting_cases['only_finetuning_correct'].append(case)
            
            elif (preds.get('finetuning', -1) != true_label and
                  any(preds[m] == true_label for m in self.model_names if m != 'finetuning')):
                interesting_cases['only_prompts_correct'].append(case)
            
            for model, conf in confs.items():
                if conf > 0.9 and preds[model] != true_label:
                    interesting_cases['high_confidence_wrong'].append({
                        **case,
                        'wrong_model': model,
                        'wrong_confidence': conf
                    })
        
        return interesting_cases
    
    def generate_detailed_analysis(self, output_dir):
        consistency = self.analyze_consistency()
        
        performance = self.analyze_model_performance()
        
        interesting = self.find_interesting_cases(consistency)
        
        stats = {
            'total_samples': len(self.test_samples),
            'consistent_correct': len(consistency['consistent_correct']),
            'consistent_wrong': len(consistency['consistent_wrong']),  
            'inconsistent': len(consistency['inconsistent']),
            'consistency_rate': len(consistency['consistent_correct']) / len(self.test_samples)
        }
        
        with open(f"{output_dir}/analysis_results.json", 'w', encoding='utf-8') as f:
            json.dump({
                'statistics': stats,
                'performance': performance,
                'consistency_analysis': consistency,
                'interesting_cases': interesting
            }, f, ensure_ascii=False, indent=2)
        
        return stats, performance, consistency, interesting
    
def main():
    predictions_file = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/predictions/all_predictions.json"
    output_dir = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/raw_data"
    
    analyzer = CaseStudyAnalyzer(predictions_file)
    analyzer.generate_detailed_analysis(output_dir)

if __name__ == "__main__":
    main()