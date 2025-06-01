#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多模型数据增强效果对比实验 - 集成层次化损失函数
支持多级别准确率分析、多模型对比和层次化损失
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import torch
torch.set_float32_matmul_precision('high')
import GPUtil
warnings.filterwarnings('ignore')

# 导入基础模块
from job_offers_classifier.job_offers_classfier_new import ChineseTransformerJobOffersClassifier, get_recommended_chinese_models
from job_offers_classifier.job_offers_utils_new import create_hierarchy_node
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# # 导入数据增强模块
# from chinese_job_data_augmentation import EnhancedJobDataProcessor
# # 导入层次化损失模块
# from hierarchical_loss import HierarchicalISCOLoss

def select_gpu_with_most_free_memory():
    """
    select gpu with most free memory
    """
    gpus = GPUtil.getGPUs()
    if not gpus:
        raise RuntimeError("No GPU found!")
    gpu_id = max(gpus, key=lambda gpu: gpu.memoryFree).id
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Selected GPU: {gpu_id}, Free Memory: {gpus[gpu_id].memoryFree} MB")

# 选择显存最多的GPU
select_gpu_with_most_free_memory()


class HierarchicalMultiModelComparison:
    """集成层次化损失的多模型对比实验"""
    
    def __init__(self, csv_path: str, max_samples: int = 12000):
        self.csv_path = csv_path
        self.max_samples = max_samples
        
        # 实验配置：包含是否使用层次化损失的对比
        self.experiment_configs = [
            # {
            #     'name': 'Baseline',
            #     'use_hierarchical_loss': False,
            #     'use_multitask_learning': False,
            #     'description': '基线模型（标准交叉熵损失）'
            # },
            # {
            #     'name': 'Hierarchical',
            #     'use_hierarchical_loss': True,
            #     'use_multitask_learning': False,
            #     # 'hierarchical_loss_weights': {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0},
            #     # 'hierarchical_loss_weights': {1: 2.0, 2: 1.5, 3: 1, 4: 0.5},
            #     'hierarchical_loss_weights': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
            #     'description': '层次化损失（单任务）'
            # },
            {
                'name': 'Multitask',
                'use_hierarchical_loss': True,
                'use_multitask_learning': True,
                'hierarchical_loss_weights': {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
                'task_weights': {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
                'description': '多任务层次化学习'
            }
        ]
        
        # 测试模型
        self.test_models = [
            'bert-base-chinese',
            # 'hfl/chinese-bert-wwm-ext',
            # 'hfl/chinese-roberta-wwm-ext'
        ]
        
        # 模型信息
        self.model_info = {
            'bert-base-chinese': {
                'name': 'Google Chinese BERT',
                'description': 'Google官方中文模型'
            },
            'hfl/chinese-bert-wwm-ext': {
                'name': 'HFL Chinese BERT-wwm',
                'description': '全词掩码预训练'
            },
            'hfl/chinese-roberta-wwm-ext': {
                'name': 'HFL Chinese RoBERTa',
                'description': 'RoBERTa架构优化'
            }
        }
        
        # 训练配置
        self.training_config = {
            'max_epochs': 8, # 默认设的是8
            'patience': 4,
            'max_seq_length': 256,
            'batch_size': 64, # 原来是16
            'learning_rate': 2e-5
        }
        
        # ISCO层级定义
        self.isco_levels = {
            1: "主要职业组",
            2: "次要职业组",
            3: "次级职业组",
            4: "基本职业组"
        }
        
        print("🔬 层次化损失多模型对比实验初始化")
        print(f"   测试模型数量: {len(self.test_models)}")
        print(f"   实验配置数量: {len(self.experiment_configs)}")
        print(f"   最大样本数: {self.max_samples}")

    def calculate_hierarchical_errors(self, y_true, y_pred):
        """计算层次化错误分析"""
        error_analysis = {
            'total_errors': 0,
            'level_1_errors': 0,
            'level_2_errors': 0,
            'level_3_errors': 0,
            'level_4_errors': 0,
            'error_examples': []
        }
        
        for true_label, pred_label in zip(y_true, y_pred):
            if true_label != pred_label:
                error_analysis['total_errors'] += 1
                
                # 确定错误级别
                error_level = 4
                for level in [1, 2, 3]:
                    if true_label[:level] != pred_label[:level]:
                        error_level = level
                        break
                
                error_analysis[f'level_{error_level}_errors'] += 1
                
                # 保存错误示例
                if len(error_analysis['error_examples']) < 10:
                    error_analysis['error_examples'].append({
                        'true': true_label,
                        'pred': pred_label,
                        'error_level': error_level
                    })
        
        return error_analysis

    def train_model_with_config(self, model_name, texts, labels, config, experiment_name, results_dir):
        """使用特定配置训练模型"""
        model_display_name = self.model_info[model_name]['name']
        config_name = config['name']
        
        print(f"\n🤖 训练: {model_display_name} - {config_name} - {experiment_name}")
        print(f"   配置: {config['description']}")
        
        # 数据划分
        train_texts, test_texts, train_labels, test_labels, level_analysis = self.safe_train_test_split_with_levels(texts, labels)
        
        if len(test_texts) < 5:
            print(f"   ❌ 测试集样本不足")
            return None
        
        # 创建层次结构
        hierarchy = self.create_isco_hierarchy_from_codes(set(labels))
        
        # 创建模型目录
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        model_dir = results_dir / f"model_{safe_model_name}_{config_name}_{experiment_name}"
        
        start_time = time.time()
        
        try:
            # 创建分类器，应用层次化配置
            classifier = ChineseTransformerJobOffersClassifier(
                model_dir=str(model_dir),
                hierarchy=hierarchy,
                transformer_model=model_name,
                learning_rate=self.training_config['learning_rate'],
                batch_size=self.training_config['batch_size'],
                max_epochs=self.training_config['max_epochs'],
                early_stopping=True,
                early_stopping_patience=self.training_config['patience'],
                max_sequence_length=self.training_config['max_seq_length'],
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision=16 if torch.cuda.is_available() else 32,
                # 层次化损失配置
                use_hierarchical_loss=config.get('use_hierarchical_loss', False),
                use_multitask_learning=config.get('use_multitask_learning', False),
                hierarchical_loss_weights=config.get('hierarchical_loss_weights', None),
                task_weights=config.get('task_weights', None),
                verbose=True
            )
            
            # 准备验证集
            val_size = min(200, len(test_texts) // 3)
            
            if val_size > 0:
                val_texts = test_texts[:val_size]
                val_labels = test_labels[:val_size]
                final_test_texts = test_texts[val_size:]
                final_test_labels = test_labels[val_size:]
            else:
                val_texts, val_labels = [], []
                final_test_texts = test_texts
                final_test_labels = test_labels
            
            print(f"   训练样本: {len(train_texts)}")
            print(f"   验证样本: {val_size if val_size > 0 else 0}")
            print(f"   测试样本: {len(final_test_texts)}")
            
            # 训练
            print(f"🎯 开始训练...")
            
            classifier.fit(train_labels, train_texts, y_val=val_labels, X_val=val_texts)
            
            # 预测
            print(f"🔮 预测中...")
            predictions_df = classifier.predict(final_test_texts, format='dataframe', top_k=5)
            
            # 提取预测结果
            y_true = final_test_labels
            y_pred = predictions_df['class_1'].tolist()
            
            # 计算各种指标
            accuracy = accuracy_score(y_true, y_pred)
            
            # 层次化准确率
            hierarchical_acc = self.calculate_hierarchical_accuracy(y_true, y_pred, None)
            
            # 错误分析
            error_analysis = self.calculate_hierarchical_errors(y_true, y_pred)
            
            # Top-k准确率
            top_3_acc = sum(
                true_label in [predictions_df.iloc[i][f'class_{j}'] for j in range(1, 4)]
                for i, true_label in enumerate(y_true)
            ) / len(y_true)
            
            top_5_acc = sum(
                true_label in [predictions_df.iloc[i][f'class_{j}'] for j in range(1, 6)]
                for i, true_label in enumerate(y_true)
            ) / len(y_true)
            
            training_time = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'model_display_name': model_display_name,
                'config_name': config_name,
                'config_description': config['description'],
                'experiment_name': experiment_name,
                'train_samples': len(train_texts),
                'test_samples': len(final_test_texts),
                'accuracy': accuracy,
                'top_3_accuracy': top_3_acc,
                'top_5_accuracy': top_5_acc,
                'training_time_minutes': training_time / 60,
                'hierarchical_accuracy': hierarchical_acc,
                'error_analysis': error_analysis,
                'status': 'success'
            }
            
            print(f"✅ 训练完成!")
            print(f"   4级准确率: {accuracy:.4f}")
            print(f"   层次化错误分布:")
            for level in [1, 2, 3, 4]:
                level_errors = error_analysis[f'level_{level}_errors']
                print(f"     {level}级错误: {level_errors} ({level_errors/len(y_true)*100:.2f}%)")
            
            return result
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'config_name': config_name,
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': str(e)
            }

    def run_hierarchical_comparison(self):
        """运行层次化损失对比实验"""
        print("🔬 开始层次化损失多模型对比实验")
        print("=" * 80)
        
        # 创建结果目录
        results_dir = Path("hierarchical_loss_comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        experiment_start_time = time.time()
        all_results = []
        
        # 加载数据
        print(f"\n📊 数据准备阶段")
        print("=" * 80)
        
        # 只加载原始数据进行对比
        texts, labels, stats = self.load_data()
        
        # 对每个模型和配置进行实验
        for model_idx, model_name in enumerate(self.test_models):
            model_display_name = self.model_info[model_name]['name']
            
            print(f"\n{'='*80}")
            print(f"🤖 模型 {model_idx+1}/{len(self.test_models)}: {model_display_name}")
            print(f"{'='*80}")
            
            for config in self.experiment_configs:
                result = self.train_model_with_config(
                    model_name, texts, labels, config, "Original", results_dir
                )
                
                if result:
                    all_results.append(result)
        
        total_time = time.time() - experiment_start_time
        
        # 生成综合报告
        self.generate_hierarchical_report(all_results, total_time, results_dir)
        
        return all_results, results_dir

    def generate_hierarchical_report(self, results, total_time, results_dir):
        """生成层次化损失对比报告"""
        print(f"\n📊 生成层次化损失对比报告...")
        
        # 保存原始结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "hierarchical_comparison_results.csv", index=False, encoding='utf-8')
        
        # 分析结果
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            print("❌ 没有成功的实验结果")
            return
        
        # 创建对比表格
        comparison_data = []
        
        for result in successful_results:
            if result['status'] == 'success':
                error_analysis = result['error_analysis']
                
                row = {
                    'Model': result['model_display_name'],
                    'Config': result['config_name'],
                    'Accuracy': result['accuracy'],
                    'Top3_Acc': result['top_3_accuracy'],
                    'Top5_Acc': result['top_5_accuracy'],
                    'Level1_Errors': error_analysis['level_1_errors'],
                    'Level2_Errors': error_analysis['level_2_errors'],
                    'Level3_Errors': error_analysis['level_3_errors'],
                    'Level4_Errors': error_analysis['level_4_errors'],
                    'Total_Errors': error_analysis['total_errors'],
                    'Training_Time': result['training_time_minutes']
                }
                
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(results_dir / "hierarchical_comparison_table.csv", index=False)
        
        # 创建可视化
        self.create_hierarchical_visualizations(comparison_df, results_dir)
        
        # 打印总结
        print(f"\n🎉 层次化损失对比实验完成!")
        print(f"📈 实验总结:")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        print(f"   测试模型: {len(self.test_models)} 个")
        print(f"   实验配置: {len(self.experiment_configs)} 种")
        print(f"   成功实验: {len(successful_results)}/{len(results)}")
        
        # 最佳配置分析
        best_accuracy = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        least_l1_errors = comparison_df.loc[comparison_df['Level1_Errors'].idxmin()]
        
        print(f"\n🏆 最佳结果:")
        print(f"   最高准确率: {best_accuracy['Model']} - {best_accuracy['Config']} ({best_accuracy['Accuracy']:.4f})")
        print(f"   最少1级错误: {least_l1_errors['Model']} - {least_l1_errors['Config']} ({least_l1_errors['Level1_Errors']}个)")
        
        print(f"\n📁 详细结果保存在: {results_dir}")

    def create_hierarchical_visualizations(self, df, results_dir):
        """创建层次化损失对比可视化"""
        # 设置matplotlib中文支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 错误级别分布对比
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 按配置分组
        configs = df['Config'].unique()
        models = df['Model'].unique()
        
        # 绘制每个级别的错误对比
        for i, level in enumerate([1, 2, 3, 4]):
            ax = axes[i//2, i%2]
            
            # 准备数据
            error_data = []
            for config in configs:
                config_data = df[df['Config'] == config]
                errors = config_data[f'Level{level}_Errors'].values
                error_data.append(errors)
            
            # 创建柱状图
            x = np.arange(len(models))
            width = 0.25
            
            for j, (config, data) in enumerate(zip(configs, error_data)):
                offset = (j - len(configs)/2) * width + width/2
                ax.bar(x + offset, data, width, label=config, alpha=0.8)
            
            ax.set_xlabel('模型', fontsize=12)
            ax.set_ylabel(f'{level}级错误数', fontsize=12)
            ax.set_title(f'{level}级分类错误对比\n({self.isco_levels[level]})', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels([m.replace(' ', '\n') for m in models], fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "hierarchical_errors_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 准确率对比热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建数据透视表
        pivot_data = df.pivot(index='Model', columns='Config', values='Accuracy')
        
        # 绘制热力图
        sns.heatmap(pivot_data, annot=True, fmt='.4f', cmap='RdYlGn', 
                    center=pivot_data.mean().mean(), ax=ax, cbar_kws={'label': '准确率'})
        
        ax.set_title('不同配置下的模型准确率对比', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('实验配置', fontsize=12)
        ax.set_ylabel('模型', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(results_dir / "accuracy_heatmap_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 错误分布雷达图
        fig, axes = plt.subplots(1, len(configs), figsize=(6*len(configs), 6))
        if len(configs) == 1:
            axes = [axes]
        
        for idx, config in enumerate(configs):
            ax = axes[idx]
            
            # 准备数据
            config_data = df[df['Config'] == config]
            
            # 设置雷达图
            categories = ['1级错误', '2级错误', '3级错误', '4级错误']
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax = plt.subplot(1, len(configs), idx+1, projection='polar')
            
            for _, row in config_data.iterrows():
                values = [
                    row['Level1_Errors'],
                    row['Level2_Errors'],
                    row['Level3_Errors'],
                    row['Level4_Errors']
                ]
                
                # 归一化到0-1范围
                max_errors = max(values) if max(values) > 0 else 1
                values = [v/max_errors for v in values]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_title(f'{config}配置\n错误分布', fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
            ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(results_dir / "error_distribution_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 训练时间vs准确率散点图
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 为每个配置使用不同的标记
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
        
        for i, config in enumerate(configs):
            config_data = df[df['Config'] == config]
            ax.scatter(config_data['Training_Time'], config_data['Accuracy'], 
                      label=config, marker=markers[i % len(markers)], 
                      s=100, alpha=0.7, color=colors[i])
            
            # 添加模型标签
            for _, row in config_data.iterrows():
                ax.annotate(row['Model'].split()[-1], 
                           (row['Training_Time'], row['Accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('训练时间 (分钟)', fontsize=12)
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_title('训练效率对比', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_dir / "training_efficiency_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 层次化对比可视化已生成:")
        print(f"   - hierarchical_errors_comparison.png: 各级别错误对比")
        print(f"   - accuracy_heatmap_comparison.png: 准确率热力图")
        print(f"   - error_distribution_radar.png: 错误分布雷达图")
        print(f"   - training_efficiency_comparison.png: 训练效率对比")

    def create_isco_hierarchy_from_codes(self, isco_codes):
        """从ISCO编码创建层次结构"""
        hierarchy = {}
        
        for code in isco_codes:
            code_str = str(code).zfill(4)
            
            for level in [1, 2, 3, 4]:
                level_code = code_str[:level]
                if level_code not in hierarchy:
                    hierarchy[level_code] = create_hierarchy_node(
                        level_code, 
                        f"ISCO-{level}位-{level_code}"
                    )
        
        return hierarchy

    def load_data(self):
        """加载数据"""
        print(f"\n📊 加载数据...")
        
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding='gbk')
        
        # 基础文本组合
        def combine_features(row):
            parts = []
            for col in ['岗位', '岗位描述', '岗位职能']:
                if col in row and pd.notna(row[col]):
                    content = str(row[col])
                    if col == '岗位职能' and content.startswith('['):
                        try:
                            import ast
                            job_funcs = ast.literal_eval(content)
                            content = ' '.join(job_funcs) if isinstance(job_funcs, list) else content
                        except:
                            pass
                    parts.append(content)
            return ' '.join(parts)
        
        df['combined_text'] = df.apply(combine_features, axis=1)
        df['isco_code'] = df['ISCO_4_Digit_Code_Gemini'].astype(str).str.zfill(4)
        
        # 移除空文本
        df = df[df['combined_text'].str.strip() != ''].copy()
        
        # 采样
        if self.max_samples and len(df) > self.max_samples:
            df = df.sample(n=self.max_samples, random_state=42)
        
        texts = df['combined_text'].tolist()
        labels = df['isco_code'].tolist()
        
        # 统计信息
        stats = {
            'total_samples': len(texts),
            'unique_labels': len(set(labels)),
            'avg_text_length': np.mean([len(text) for text in texts])
        }
        
        print(f"✅ 数据加载完成:")
        print(f"   样本数: {len(texts)}")
        print(f"   类别数: {len(set(labels))}")
        
        return texts, labels, stats

    def calculate_hierarchical_accuracy(self, y_true, y_pred, top_k_preds=None):
        """计算层次化准确率"""
        results = {}
        
        for level in [1, 2, 3, 4]:
            true_level = [label[:level] for label in y_true]
            pred_level = [label[:level] for label in y_pred]
            
            accuracy = accuracy_score(true_level, pred_level)
            results[f'level_{level}_accuracy'] = accuracy
        
        return results

    def safe_train_test_split_with_levels(self, texts, labels):
        """安全的数据划分"""
        label_counts = Counter(labels)
        
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        train_indices = []
        test_indices = []
        
        # 单样本类别全部放入训练集
        for i, (text, label) in enumerate(zip(texts, labels)):
            if label in single_sample_classes:
                train_indices.append(i)
        
        # 多样本类别分层划分
        if multi_sample_classes:
            multi_texts = []
            multi_labels = []
            multi_indices = []
            
            for i, (text, label) in enumerate(zip(texts, labels)):
                if label in multi_sample_classes:
                    multi_texts.append(text)
                    multi_labels.append(label)
                    multi_indices.append(i)
            
            if len(multi_texts) > 0:
                try:
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_texts)), 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=multi_labels
                    )
                except:
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_texts)), 
                        test_size=0.2, 
                        random_state=42
                    )
                
                train_indices.extend([multi_indices[i] for i in multi_train_idx])
                test_indices.extend([multi_indices[i] for i in multi_test_idx])
        
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # 级别分析
        level_analysis = {}
        for level in [1, 2, 3, 4]:
            level_codes = [label[:level] for label in labels]
            level_counts = Counter(level_codes)
            level_analysis[level] = {
                'total_classes': len(level_counts),
                'single_sample_classes': sum(1 for c in level_counts.values() if c == 1)
            }
        
        return train_texts, test_texts, train_labels, test_labels, level_analysis


def main():
    """主函数"""
    print("🔬 层次化损失多模型对比实验")
    print("=" * 80)
    
    # 配置实验
    csv_path = "./newjob1_sortall.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        return
    
    print("实验说明:")
    print("本实验将对比三种损失函数配置:")
    print("1. Baseline: 标准交叉熵损失")
    print("2. Hierarchical: 层次化损失（根据错误级别给予不同惩罚）")
    print("3. Multitask: 多任务层次化学习（同时预测1-4级）")
    
    # confirm = input("\n确认开始实验? (y/N): ")
    # if confirm.lower() != 'y':
    #     print("实验已取消")
    #     return
    
    try:
        # 创建实验对象
        experiment = HierarchicalMultiModelComparison(csv_path, max_samples=4000)
        
        # 运行对比实验
        results, results_dir = experiment.run_hierarchical_comparison()
        
        print(f"\n🎯 实验完成!")
        print(f"📁 查看详细结果: {results_dir}")
        
    except Exception as e:
        print(f"❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()