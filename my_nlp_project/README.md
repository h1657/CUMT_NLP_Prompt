# 少样本情感分类实验项目

本项目基于 [OpenPrompt](https://github.com/thunlp/OpenPrompt) 框架。项目全面对比了标准微调与多种提示学习方法在不同设置下的性能与效率。

## 环境准备

1.  **安装依赖:**
    
    ```bash
    pip install -r ../requirements.txt
    ```
    
3.  **准备预训练模型:**
    请提前下载 `bert-base-uncased` 模型。下载方式：[google-bert/bert-base-uncased at main](https://hf-mirror.com/google-bert/bert-base-uncased/tree/main)

4.  **修改配置路径:**
    本项目的所有配置文件（位于 `my_nlp_project/configs/`）中的模型和文件路径均为绝对路径。在运行前，请**务必**批量替换所有`.yaml`文件中的根路径 `/root/autodl-tmp/OpenPrompt/` 为您自己的项目根目录绝对路径。

## 实验流程

### 第一步：运行训练实验

所有实验均通过项目根目录下的 `experiments/cli.py` 脚本启动，通过加载不同的配置文件来执行。

- **提示学习实验 (Prompt-tuning):**
  
  ```bash
  # 示例：运行参数高效的 Soft Template (Freeze) 实验
  python ../experiments/cli.py --config_yaml my_nlp_project/configs/sst2_soft_template_freeze.yaml
  
  # 示例：运行人工模板实验 (16-shot)
  python ../experiments/cli.py --config_yaml my_nlp_project/configs/sst2_manual_prompt.yaml
  ```
  > **说明**: 更换 `--config_yaml` 参数即可运行 `my_nlp_project/configs/` 目录下的其他实验配置。
  
- **标准微调实验 (Fine-tuning):**
  本项目中的标准微调是在全量数据上进行的，可使用 `my_nlp_project/scripts/run_finetuning.py` 脚本（需先修改脚本中的硬编码路径为您的实际路径）。

### 第二步：结果分析与可视化

分析脚本均位于 `my_nlp_project/scripts/analysis/` 目录。请在所有训练实验完成后，按需执行。**注意：** 同样需要将脚本内硬编码的路径 `/root/autodl-tmp/OpenPrompt/` 替换为您的项目根目录。

1.  **汇总所有模型预测 (用于案例分析):**
    ```bash
    python my_nlp_project/scripts/analysis/run_inference.py
    ```
    - **生成文件**: `my_nlp_project/outputs/predictions/all_predictions.json`

2.  **收集主要性能指标:**
    
    ```bash
    python my_nlp_project/scripts/analysis/collect_results.py
    ```
    - **生成文件**: `my_nlp_project/outputs/raw_data/complete_results.json`
    
3.  **执行定性案例分析:**
    
    ```bash
    python my_nlp_project/scripts/analysis/perform_case_study.py
    ```
    - **生成文件**: `my_nlp_project/outputs/raw_data/analysis_results.json`
    
4.  **分析软提示的语义:**
    ```bash
    # 步骤1: 提取软提示向量
    python my_nlp_project/scripts/analysis/soft_prompt_analyzer.py
    # 步骤2: 计算与词汇表的相似度
    python my_nlp_project/scripts/analysis/similarity_calculator.py
    # 步骤3: 生成相似度图表
    python my_nlp_project/scripts/analysis/sim_visualization.py
    ```
    - **生成文件**: 
      - `my_nlp_project/outputs/embeddings/` (软提示向量)
      - `my_nlp_project/outputs/similarity_matrices/` (相似度计算结果)
      - `my_nlp_project/results/figures/` (相似度图表)
    
5.  **生成报告核心图表:**
    
    ```bash
    python my_nlp_project/scripts/analysis/visualize.py
    ```
    - **生成文件**: `my_nlp_project/results/figures/`

## 预期完整目录结构

运行完所有脚本后，完整的项目目录结构如下：

```
my_nlp_project/
├── README.md
├── configs/                               # 实验配置文件
│   ├── sst2_manual_prompt.yaml
│   ├── sst2_manual_prompt_8shot.yaml
│   ├── sst2_manual_prompt_32shot.yaml
│   ├── sst2_proto_verbalizer.yaml
│   ├── sst2_ptuning.yaml
│   ├── sst2_ptuning_freeze.yaml
│   ├── sst2_soft_prompt.yaml
│   ├── sst2_soft_template_freeze.yaml
│   └── sst2_soft_verbalizer.yaml
├── data/                                  # 数据集
│   └── SST-2/
│       ├── train.tsv
│       ├── dev.tsv
│       └── test.tsv
├── logs/                                  # 训练日志和模型检查点
│   ├── sst2_bert_manual_template_manual_verbalizer_16shot/
│   ├── sst2_bert_manual_template_manual_verbalizer_8shot/
│   ├── sst2_bert_manual_template_manual_verbalizer_32shot/
│   ├── sst2_bert_manual_template_proto_verbalizer_16shot/
│   ├── sst2_bert_manual_template_soft_verbalizer_16shot/
│   ├── sst2_bert_ptuning_freeze_16shot/
│   ├── sst2_bert_ptuning_template_manual_verbalizer_16shot/
│   ├── sst2_bert_soft_template_freeze_16shot/
│   └── sst2_bert_soft_template_manual_verbalizer_16shot/
├── prompt_resources/                      # 提示模板和标签词
│   └── SST-2/
│       ├── manual_template.txt
│       ├── manual_verbalizer.txt
│       ├── ptuning_template.txt
│       └── soft_template.txt
├── outputs/                               # 分析输出
│   ├── predictions/
│   │   └── all_predictions.json          # 所有模型的预测结果
│   ├── raw_data/
│   │   ├── complete_results.json         # 完整性能统计
│   │   └── analysis_results.json         # 案例分析结果
│   ├── embeddings/                       # 软提示向量文件
│   │   ├── sst2_bert_soft_template_freeze_embeddings.pt
│   │   └── sst2_bert_ptuning_freeze_embeddings.pt
│   └── similarity_matrices/              # 相似度分析结果
│       ├── sst2_bert_soft_template_freeze_similarity.json
│       └── sst2_bert_ptuning_freeze_similarity.json
├── results/                              # 可视化图表
│   └── figures/
│       ├── consistency_distribution_en.png
│       ├── error_analysis_en.png
│       ├── parameter_efficiency_comparison_en.png
│       ├── performance_comparison_en.png
│       ├── sst2_bert_ptuning_freeze_heatmap.png
│       ├── sst2_bert_ptuning_freeze_top_words.png
│       ├── sst2_bert_soft_template_freeze_heatmap.png
│       ├── sst2_bert_soft_template_freeze_top_words.png
│       └── sst2_bert_soft_template_freeze_vs_sst2_bert_ptuning_freeze_comparison.png
└── scripts/                              # 分析脚本
    ├── run_finetuning.py                 # 标准微调脚本
    └── analysis/
        ├── calc_parameter_efficiency.py
        ├── collect_results.py
        ├── perform_case_study.py
        ├── run_inference.py
        ├── similarity_calculator.py
        ├── sim_visualization.py
        ├── soft_prompt_analyzer.py
        └── visualize.py
```