#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties

class ResultVisualizer:
    def __init__(self, analysis_file):
        with open(analysis_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
    def plot_consistency_distribution(self, output_dir):
        stats = self.data['statistics']
        
        labels = ['Consistent & Correct', 'Consistent & Wrong', 'Inconsistent']
        sizes = [stats['consistent_correct'], stats['consistent_wrong'], stats['inconsistent']]
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        
        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Model Prediction Consistency Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/consistency_distribution_en.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_comparison(self, output_dir):
        performance = self.data['performance']
        
        models = list(performance.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        x = np.arange(len(models))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [performance[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=name)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_comparison_en.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_error_analysis(self, output_dir):
        interesting = self.data['interesting_cases']
        
        error_types = ['Correct only by Fine-tuning', 'Correct only by Prompting', 'High-Confidence Errors']
        counts = [
            len(interesting['only_finetuning_correct']),
            len(interesting['only_prompts_correct']), 
            len(interesting['high_confidence_wrong'])
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(error_types, counts, color=['#3498db', '#9b59b6', '#e67e22'])
        plt.title('Distribution of Different Error Case Types', fontsize=16, fontweight='bold')
        plt.ylabel('Number of Cases', fontsize=12)
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_analysis_en.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_all_visualizations(self, output_dir):
        self.plot_consistency_distribution(output_dir)
        self.plot_performance_comparison(output_dir)
        self.plot_error_analysis(output_dir)

def main():
    analysis_file = "/root/autodl-tmp/OpenPrompt/my_nlp_project/outputs/raw_data/analysis_results.json"
    output_dir = "/root/autodl-tmp/OpenPrompt/my_nlp_project/results/figures"
    
    visualizer = ResultVisualizer(analysis_file)
    visualizer.generate_all_visualizations(output_dir)

if __name__ == "__main__":
    main()