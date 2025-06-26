#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np
import pickle
from pathlib import Path
from transformers import BertTokenizer, BertModel
import pandas as pd
from tqdm import tqdm
import time

class SimilarityCalculator:
    def __init__(self, project_root='/root/autodl-tmp/OpenPrompt/my_nlp_project'):
        self.project_root = Path(project_root)
        self.results_dir = self.project_root / 'outputs'
        self.tokenizer = BertTokenizer.from_pretrained('/root/autodl-tmp/OpenPrompt/bert-base-uncased')
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.bert_model = BertModel.from_pretrained('/root/autodl-tmp/OpenPrompt/bert-base-uncased')
        self.vocab_embeddings = self.bert_model.embeddings.word_embeddings.weight.detach().to(self.device)
        self.embed_dim = self.vocab_embeddings.shape[1]
        
    def load_soft_embeddings(self, experiment_name):
        embedding_path = self.results_dir / 'embeddings' / f'{experiment_name}_embeddings.pkl'
        with open(embedding_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
        
    def calculate_vocabulary_similarity_optimized(self, soft_embeddings, top_k=10, batch_size=32):
        similarities = {}
        
        total_tokens = 0
        valid_params = {}
        
        for param_name, embedding in soft_embeddings.items():
            if len(embedding.shape) == 2:
                if embedding.shape[1] == self.embed_dim:
                    total_tokens += embedding.shape[0]
                    valid_params[param_name] = embedding
        
        if total_tokens == 0:
            return {}
        
        overall_pbar = tqdm(total=total_tokens, desc="总体进度", position=0)
        
        for param_name, embedding in valid_params.items():
            param_similarities = []
            
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            num_tokens = embedding_tensor.shape[0]
            
            param_pbar = tqdm(total=num_tokens, desc=f"参数 {param_name}", position=1, leave=False)
            
            for start_idx in range(0, num_tokens, batch_size):
                end_idx = min(start_idx + batch_size, num_tokens)
                batch_embeddings = embedding_tensor[start_idx:end_idx]
                
                with torch.no_grad():
                    try:
                        cos_sim = F.cosine_similarity(
                            batch_embeddings.unsqueeze(1),
                            self.vocab_embeddings.unsqueeze(0),
                            dim=2
                        )
                        
                        top_similarities, top_indices = torch.topk(cos_sim, top_k, dim=1)
                        
                        top_similarities = top_similarities.cpu().numpy()
                        top_indices = top_indices.cpu().numpy()
                        
                    except RuntimeError as e:
                        raise e
                
                for i in range(end_idx - start_idx):
                    token_index = start_idx + i
                    token_top_indices = top_indices[i]
                    token_similarities = top_similarities[i]
                    
                    top_words = [self.tokenizer.convert_ids_to_tokens([idx])[0] for idx in token_top_indices]
                    
                    param_similarities.append({
                        'token_index': token_index,
                        'top_words': top_words,
                        'similarities': token_similarities.tolist()
                    })
                    
                    param_pbar.update(1)
                    overall_pbar.update(1)
            
            param_pbar.close()
            similarities[param_name] = param_similarities
            
            del embedding_tensor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        overall_pbar.close()
        return similarities
    
    def calculate_vocabulary_similarity(self, soft_embeddings, top_k=10):
        return self.calculate_vocabulary_similarity_optimized(soft_embeddings, top_k)
        
    def analyze_embeddings_structure(self, soft_embeddings):
        for param_name, embedding in soft_embeddings.items():
            if 'ptuning' in param_name:
                display_name = "ptuning_combined_embeddings"
            else:
                display_name = param_name

            shape_str = f"形状: {embedding.shape}"
            if len(embedding.shape) == 2:
                if embedding.shape[1] == self.embed_dim:
                    status = "维度匹配"
                else:
                    status = f"维度不匹配 (期望{self.embed_dim}, 实际{embedding.shape[1]})"
            else:
                status = "非2D张量，跳过"
        
    def save_similarity_results(self, experiment_name, similarities):
        if not similarities:
            return
            
        save_path = self.results_dir / 'similarity_matrices' / f'{experiment_name}_similarities.pkl'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(similarities, f)
            
        csv_path = self.results_dir / 'similarity_matrices' / f'{experiment_name}_similarities.csv'
        self.export_to_csv(similarities, csv_path)
        
    def export_to_csv(self, similarities, csv_path):
        rows = []
        for param_name, param_similarities in similarities.items():
            for token_data in param_similarities:
                for i, (word, sim) in enumerate(zip(token_data['top_words'], token_data['similarities'])):
                    rows.append({
                        'parameter': param_name,
                        'token_index': token_data['token_index'],
                        'rank': i + 1,
                        'similar_word': word,
                        'similarity': sim
                    })
                    
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    start_time = time.time()
    calculator = SimilarityCalculator()
    
    soft_embeddings = calculator.load_soft_embeddings('sst2_bert_soft_template_freeze')
    calculator.analyze_embeddings_structure(soft_embeddings)
    soft_similarities = calculator.calculate_vocabulary_similarity(soft_embeddings)
    calculator.save_similarity_results('sst2_bert_soft_template_freeze', soft_similarities)
    
    ptuning_embeddings = calculator.load_soft_embeddings('sst2_bert_ptuning_freeze')
    calculator.analyze_embeddings_structure(ptuning_embeddings)
    ptuning_similarities = calculator.calculate_vocabulary_similarity(ptuning_embeddings)
    calculator.save_similarity_results('sst2_bert_ptuning_freeze', ptuning_similarities)
    
    total_time = time.time() - start_time