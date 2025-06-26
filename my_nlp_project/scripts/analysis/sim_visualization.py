#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class VisualizationGenerator:
    def __init__(self, project_root='/root/autodl-tmp/OpenPrompt/my_nlp_project'):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'outputs'
        
    def load_similarity_data(self, experiment_name):
        pkl_path = self.results_dir / 'similarity_matrices' / f'{experiment_name}_similarities.pkl'
        csv_path = self.results_dir / 'similarity_matrices' / f'{experiment_name}_similarities.csv'
        
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
            
        csv_data = pd.read_csv(csv_path)
        return pkl_data, csv_data
        
    def plot_top_similar_words(self, experiment_name, num_tokens=5):
        _, csv_data = self.load_similarity_data(experiment_name)
        
        if 'ptuning' in experiment_name:
            param_name = 'ptuning_combined_embeddings'
        else:
            param_name = csv_data['parameter'].unique()[0]
            
        param_data = csv_data[csv_data['parameter'] == param_name]
        
        filtered_data = param_data[
            (param_data['token_index'] < num_tokens) & 
            (param_data['rank'] <= 5)
        ]
        
        fig, axes = plt.subplots(1, num_tokens, figsize=(4 * num_tokens, 5))
        if num_tokens == 1:
            axes = [axes]
            
        for i in range(num_tokens):
            token_data = filtered_data[filtered_data['token_index'] == i]
            
            if not token_data.empty:
                words = token_data['similar_word'].tolist()
                similarities = token_data['similarity'].tolist()
                
                axes[i].barh(range(len(words)), similarities)
                axes[i].set_yticks(range(len(words)))
                axes[i].set_yticklabels(words)
                axes[i].set_xlabel('Cosine Similarity')
                axes[i].set_title(f'Token {i} Top Words')
                axes[i].set_xlim(0, 1)
                
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.6)
        save_path = '/root/autodl-tmp/OpenPrompt/my_nlp_project/results/figures' / f'{experiment_name}_top_words.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_similarity_heatmap(self, experiment_name, max_tokens=10):
        _, csv_data = self.load_similarity_data(experiment_name)
        
        if 'ptuning' in experiment_name:
            param_name = 'ptuning_combined_embeddings'
        else:
            param_name = csv_data['parameter'].unique()[0]
        
        filtered_data = csv_data[
            (csv_data['parameter'] == param_name) &
            (csv_data['token_index'] < max_tokens) & 
            (csv_data['rank'] == 1)
        ]
        
        pivot_data = filtered_data.pivot_table(
            values='similarity', 
            index='token_index', 
            columns='similar_word',
            fill_value=0
        )
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(pivot_data, annot=True, cmap='Blues', fmt='.3f')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'{experiment_name} - Soft Prompt Token Similarity Heatmap')
        plt.xlabel('Most Similar Words (Rank 1)')
        plt.ylabel('Soft Prompt Token Index')
        
        plt.tight_layout()
        save_path = '/root/autodl-tmp/OpenPrompt/my_nlp_project/results/figures' / f'{experiment_name}_heatmap.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def compare_experiments(self, exp1, exp2, num_tokens=5):
        _, csv1 = self.load_similarity_data(exp1)
        _, csv2 = self.load_similarity_data(exp2)
        
        top1 = csv1[(csv1['rank'] == 1) & (csv1['token_index'] < num_tokens)]
        top2 = csv2[(csv2['rank'] == 1) & (csv2['token_index'] < num_tokens)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(8, 1.2 * num_tokens)))
        
        tokens1 = top1['token_index'].tolist()
        similarities1 = top1['similarity'].tolist()
        words1 = top1['similar_word'].tolist()
        
        ax1.barh(range(len(tokens1)), similarities1)
        ax1.set_yticks(range(len(tokens1)))
        ax1.set_yticklabels([f"Token {i}: {w}" for i, w in zip(tokens1, words1)])
        ax1.set_xlabel('Cosine Similarity')
        ax1.set_title(f'{exp1}')
        
        tokens2 = top2['token_index'].tolist()
        similarities2 = top2['similarity'].tolist()
        words2 = top2['similar_word'].tolist()
        
        ax2.barh(range(len(tokens2)), similarities2)
        ax2.set_yticks(range(len(tokens2)))
        ax2.set_yticklabels([f"Token {i}: {w}" for i, w in zip(tokens2, words2)])
        ax2.set_xlabel('Cosine Similarity')
        ax2.set_title(f'{exp2}')
        
        plt.tight_layout()
        save_path = '/root/autodl-tmp/OpenPrompt/my_nlp_project/results/figures' / f'{exp1}_vs_{exp2}_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    viz = VisualizationGenerator()
    
    viz.plot_top_similar_words('sst2_bert_soft_template_freeze', num_tokens=5)
    viz.plot_similarity_heatmap('sst2_bert_soft_template_freeze', max_tokens=8)
    
    viz.plot_top_similar_words('sst2_bert_ptuning_freeze', num_tokens=5)
    viz.plot_similarity_heatmap('sst2_bert_ptuning_freeze', max_tokens=5)
    
    viz.compare_experiments('sst2_bert_soft_template_freeze', 'sst2_bert_ptuning_freeze', num_tokens=5)