#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版多模型数据增强效果对比实验
支持多级别准确率分析和多模型对比
"""

import os
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

warnings.filterwarnings('ignore')

# 导入原有模块
if __name__ == '__main__':
    # 添加路径以便导入lunwenimpro下的模块
    import sys
    sys.path.append('.')  # 当前目录
    sys.path.append('lunwenimpro')  # lunwenimpro目录
    
    from lunwenimpro.job_offers_classifier.job_offers_classfier import (
        ChineseTransformerJobOffersClassifier,
        get_recommended_chinese_models
    )
    from lunwenimpro.job_offers_classifier.job_offers_utils import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    # 导入数据增强模块 - 确保这个文件在当前目录下
    from chinese_job_data_augmentation import EnhancedJobDataProcessor
    from job_offers_classifier.job_offers_utils_old import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    # 导入数据增强模块
    from chinese_job_data_augmentation import EnhancedJobDataProcessor


class EnhancedMultiModelComparison:
    """增强版多模型数据增强效果对比实验"""
    
    def __init__(self, csv_path: str, max_samples: int = 12000):
        self.csv_path = csv_path
        self.max_samples = max_samples
        
        # 多模型配置 - 使用支持的中文模型
        self.test_models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
        
        # 模型信息映射
        self.model_info = {
            'bert-base-chinese': {
                'name': 'Google Chinese BERT',
                'description': 'Google官方中文模型',
                'params': '110M'
            },
            'hfl/chinese-bert-wwm-ext': {
                'name': 'HFL Chinese BERT-wwm',
                'description': '全词掩码预训练',
                'params': '110M'
            },
            'hfl/chinese-roberta-wwm-ext': {
                'name': 'HFL Chinese RoBERTa',
                'description': 'RoBERTa架构优化',
                'params': '110M'
            },
            'ckiplab/bert-base-chinese': {
                'name': 'CKIP Chinese BERT',
                'description': '台湾中研院版本',
                'params': '110M'
            },
            'bert-base-multilingual-cased': {
                'name': 'BERT Multilingual',
                'description': '多语言基准模型',
                'params': '110M'
            }
        }
        
        # 训练配置
        self.training_config = {
            'max_epochs': 8,
            'patience': 4,
            'max_seq_length': 256,
            'batch_size': 16,
            'learning_rate': 2e-5
        }
        
        # ISCO层级定义
        self.isco_levels = {
            1: "主要职业组",
            2: "次要职业组", 
            3: "次级职业组",
            4: "基本职业组"
        }
        
        print("🔬 增强版多模型数据增强效果对比实验初始化")
        print(f"   测试模型数量: {len(self.test_models)}")
        print(f"   最大样本数: {self.max_samples}")
        print(f"   ISCO层级分析: {list(self.isco_levels.keys())}")

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

    def load_data(self, enable_augmentation=False):
        """加载数据"""
        data_type = "增强" if enable_augmentation else "原始"
        print(f"\n📊 加载{data_type}数据...")
        
        if enable_augmentation:
            # 使用增强数据
            processor = EnhancedJobDataProcessor()
            texts, labels, processing_stats = processor.process_csv_data(
                csv_path=self.csv_path,
                enable_augmentation=True,
                balance_data=True,
                target_samples_per_class=8
            )
            
            # 限制样本数
            if self.max_samples and len(texts) > self.max_samples:
                indices = np.random.choice(len(texts), size=self.max_samples, replace=False)
                texts = [texts[i] for i in indices]
                labels = [labels[i] for i in indices]
            
            stats = processing_stats['final_stats']
            
        else:
            # 使用原始数据
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
            
            # 智能采样
            if self.max_samples and len(df) > self.max_samples:
                class_counts = df['isco_code'].value_counts()
                single_classes = class_counts[class_counts == 1].index.tolist()
                multi_classes = class_counts[class_counts > 1].index.tolist()
                
                sampled_data = []
                single_sample_data = df[df['isco_code'].isin(single_classes)]
                remaining_budget = self.max_samples - len(single_sample_data)
                
                if remaining_budget > 0:
                    sampled_data.append(single_sample_data)
                    multi_sample_data = df[df['isco_code'].isin(multi_classes)]
                    
                    if len(multi_sample_data) > remaining_budget:
                        try:
                            multi_sample_data = multi_sample_data.groupby('isco_code', group_keys=False).apply(
                                lambda x: x.sample(min(len(x), max(2, remaining_budget // len(multi_classes))), 
                                                 random_state=42)
                            ).reset_index(drop=True)
                            
                            if len(multi_sample_data) > remaining_budget:
                                multi_sample_data = multi_sample_data.sample(n=remaining_budget, random_state=42)
                        except ValueError:
                            multi_sample_data = multi_sample_data.sample(n=remaining_budget, random_state=42)
                    
                    sampled_data.append(multi_sample_data)
                else:
                    sampled_data.append(single_sample_data.sample(n=self.max_samples, random_state=42))
                
                df = pd.concat(sampled_data, ignore_index=True)
            
            texts = df['combined_text'].tolist()
            labels = df['isco_code'].tolist()
            
            # 过滤单样本类别
            final_class_counts = pd.Series(labels).value_counts()
            single_classes = final_class_counts[final_class_counts == 1].index.tolist()
            
            if len(single_classes) > len(final_class_counts) * 0.3:
                multi_classes = final_class_counts[final_class_counts > 1].index.tolist()
                filtered_data = [(text, label) for text, label in zip(texts, labels) if label in multi_classes]
                
                if len(filtered_data) > 100:
                    texts, labels = zip(*filtered_data)
                    texts, labels = list(texts), list(labels)
            
            # 统计信息
            text_lengths = [len(text) for text in texts]
            stats = {
                'total_samples': len(texts),
                'unique_labels': len(set(labels)),
                'avg_text_length': np.mean(text_lengths),
                'avg_word_count': np.mean([len(text.split()) for text in texts]),
                'label_distribution': Counter(labels)
            }
        
        print(f"✅ {data_type}数据加载完成:")
        print(f"   样本数: {len(texts)}")
        print(f"   类别数: {len(set(labels))}")
        print(f"   平均文本长度: {stats['avg_text_length']:.1f}")
        
        return texts, labels, stats

    def analyze_isco_levels(self, labels):
        """分析ISCO各级别分布"""
        level_stats = {}
        
        for level in [1, 2, 3, 4]:
            level_codes = [label[:level] for label in labels]
            level_unique = len(set(level_codes))
            level_stats[level] = {
                'unique_codes': level_unique,
                'description': self.isco_levels[level],
                'codes': list(set(level_codes))
            }
        
        return level_stats

    def calculate_hierarchical_accuracy(self, y_true, y_pred, top_k_preds=None):
        """计算层次化准确率"""
        results = {}
        
        # 各级别准确率
        for level in [1, 2, 3, 4]:
            true_level = [label[:level] for label in y_true]
            pred_level = [label[:level] for label in y_pred]
            
            accuracy = accuracy_score(true_level, pred_level)
            results[f'level_{level}_accuracy'] = accuracy
            
            # Top-k层次化准确率（如果有top_k预测）
            if top_k_preds is not None:
                top_k_level_acc = []
                for i, true_code in enumerate(true_level):
                    # 检查true_code是否在top_k预测的任何一个的相应级别中
                    match_found = False
                    for pred_code in top_k_preds[i]:
                        if str(pred_code)[:level] == true_code:
                            match_found = True
                            break
                    top_k_level_acc.append(match_found)
                
                results[f'level_{level}_top5_accuracy'] = np.mean(top_k_level_acc)
        
        return results

    def safe_train_test_split_with_levels(self, texts, labels):
        """考虑层次结构的安全数据划分"""
        print("   📊 分析数据分布...")
        label_counts = Counter(labels)
        
        # 分析各级别的分布
        level_analysis = {}
        for level in [1, 2, 3, 4]:
            level_codes = [label[:level] for label in labels]
            level_counts = Counter(level_codes)
            single_sample_classes = [code for code, count in level_counts.items() if count == 1]
            level_analysis[level] = {
                'total_classes': len(level_counts),
                'single_sample_classes': len(single_sample_classes),
                'multi_sample_classes': len(level_counts) - len(single_sample_classes)
            }
        
        # 使用4级编码进行划分（最细粒度）
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        print(f"   4级编码 - 单样本类别: {len(single_sample_classes)}, 多样本类别: {len(multi_sample_classes)}")
        
        train_indices = []
        test_indices = []
        
        # 单样本类别全部放入训练集
        for i, (text, label) in enumerate(zip(texts, labels)):
            if label in single_sample_classes:
                train_indices.append(i)
        
        # 多样本类别正常分层划分
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
                except ValueError:
                    # 分层失败，使用随机划分
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_texts)), 
                        test_size=0.2, 
                        random_state=42
                    )
                
                train_indices.extend([multi_indices[i] for i in multi_train_idx])
                test_indices.extend([multi_indices[i] for i in multi_test_idx])
        
        # 构建最终的训练测试集
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # 验证测试集
        if len(test_texts) == 0 and len(train_texts) >= 10:
            split_point = max(1, len(train_texts) // 5)
            test_texts = train_texts[-split_point:]
            test_labels = train_labels[-split_point:]
            train_texts = train_texts[:-split_point]
            train_labels = train_labels[:-split_point]
        
        print(f"   最终划分 - 训练: {len(train_texts)}, 测试: {len(test_texts)}")
        
        return train_texts, test_texts, train_labels, test_labels, level_analysis

    def train_and_evaluate_model(self, model_name, texts, labels, experiment_name, results_dir):
        """训练和评估单个模型"""
        model_display_name = self.model_info[model_name]['name']
        print(f"\n🤖 开始训练: {model_display_name} ({model_name}) - {experiment_name}")
        
        # 数据划分
        train_texts, test_texts, train_labels, test_labels, level_analysis = self.safe_train_test_split_with_levels(texts, labels)
        
        if len(test_texts) < 5:
            print(f"   ❌ 测试集样本不足: {len(test_texts)}")
            return {
                'model_name': model_name,
                'model_display_name': model_display_name,
                'experiment_name': experiment_name,
                'status': 'failed',
                'error': 'Insufficient test data'
            }
        
        # 创建层次结构
        hierarchy = self.create_isco_hierarchy_from_codes(set(labels))
        
        # 创建模型目录
        safe_model_name = model_name.replace('/', '_').replace('-', '_')
        model_dir = results_dir / f"model_{safe_model_name}_{experiment_name.lower().replace(' ', '_')}"
        
        start_time = time.time()
        
        try:
            # 为不同模型优化学习率
            learning_rate = self.training_config['learning_rate']
            if 'roberta' in model_name.lower():
                learning_rate = 2.5e-5  # RoBERTa通常需要稍高的学习率
            elif 'multilingual' in model_name.lower():
                learning_rate = 1.5e-5  # 多语言模型使用较低学习率
            
            # 创建分类器
            classifier = ChineseTransformerJobOffersClassifier(
                model_dir=str(model_dir),
                hierarchy=hierarchy,
                transformer_model=model_name,
                learning_rate=learning_rate,
                batch_size=self.training_config['batch_size'],
                max_epochs=self.training_config['max_epochs'],
                early_stopping=True,
                early_stopping_patience=self.training_config['patience'],
                max_sequence_length=self.training_config['max_seq_length'],
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision="16-mixed" if torch.cuda.is_available() else 32,
                threads=0,
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
                val_texts, val_labels = None, None
                final_test_texts = test_texts
                final_test_labels = test_labels
            
            print(f"   训练样本: {len(train_texts)}")
            print(f"   验证样本: {val_size if val_size > 0 else 0}")
            print(f"   测试样本: {len(final_test_texts)}")
            print(f"   学习率: {learning_rate}")
            
            # 训练
            print(f"   🎯 开始训练...")
            classifier.fit(train_labels, train_texts, y_val=val_labels, X_val=val_texts)
            
            # 预测
            print(f"   🔮 预测中...")
            predictions_df = classifier.predict(final_test_texts, format='dataframe', top_k=5)
            
            # 提取top-k预测用于层次化分析
            top_k_predictions = []
            for i in range(len(final_test_texts)):
                row_preds = []
                for k in range(1, 6):  # top-5
                    pred = predictions_df.iloc[i][f'class_{k}']
                    if pd.notna(pred):
                        row_preds.append(pred)
                top_k_predictions.append(row_preds)
            
            # 计算基本指标
            y_true = final_test_labels
            y_pred = predictions_df['class_1'].tolist()
            
            accuracy = accuracy_score(y_true, y_pred)
            top_3_acc = sum(
                true_label in [predictions_df.iloc[i][f'class_{j}'] for j in range(1, 4)]
                for i, true_label in enumerate(y_true)
            ) / len(y_true)
            top_5_acc = sum(
                true_label in [predictions_df.iloc[i][f'class_{j}'] for j in range(1, 6)]
                for i, true_label in enumerate(y_true)
            ) / len(y_true)
            
            # 计算层次化准确率
            hierarchical_results = self.calculate_hierarchical_accuracy(y_true, y_pred, top_k_predictions)
            
            # 分析ISCO级别分布
            test_level_stats = self.analyze_isco_levels(final_test_labels)
            
            training_time = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'model_display_name': model_display_name,
                'model_description': self.model_info[model_name]['description'],
                'experiment_name': experiment_name,
                'train_samples': len(train_texts),
                'test_samples': len(final_test_texts),
                'learning_rate': learning_rate,
                'accuracy': accuracy,
                'top_3_accuracy': top_3_acc,
                'top_5_accuracy': top_5_acc,
                'training_time_minutes': training_time / 60,
                'status': 'success',
                'hierarchical_accuracy': hierarchical_results,
                'level_analysis': {
                    'training_distribution': level_analysis,
                    'test_distribution': test_level_stats
                }
            }
            
            print(f"✅ {model_display_name} - {experiment_name} 训练完成!")
            print(f"   4级准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   3级准确率: {hierarchical_results['level_3_accuracy']:.4f}")
            print(f"   2级准确率: {hierarchical_results['level_2_accuracy']:.4f}")
            print(f"   1级准确率: {hierarchical_results['level_1_accuracy']:.4f}")
            print(f"   训练时间: {training_time/60:.1f} 分钟")
            
            # 保存详细结果
            detailed_results = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'confidence': predictions_df['prob_1'].tolist(),
                'correct': [t == p for t, p in zip(y_true, y_pred)],
                'true_level_1': [label[:1] for label in y_true],
                'pred_level_1': [label[:1] for label in y_pred],
                'true_level_2': [label[:2] for label in y_true],
                'pred_level_2': [label[:2] for label in y_pred],
                'true_level_3': [label[:3] for label in y_true],
                'pred_level_3': [label[:3] for label in y_pred]
            })
            
            detailed_results.to_csv(
                results_dir / f"{safe_model_name}_{experiment_name.lower().replace(' ', '_')}_detailed.csv", 
                index=False, encoding='utf-8'
            )
            
            return result
            
        except Exception as e:
            print(f"❌ {model_display_name} - {experiment_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'model_display_name': model_display_name,
                'experiment_name': experiment_name,
                'accuracy': 0.0,
                'training_time_minutes': 0.0,
                'status': 'failed',
                'error': str(e)
            }

    def run_comprehensive_comparison(self):
        """运行全面的多模型对比实验"""
        print("🔬 开始增强版多模型数据增强效果对比实验")
        print("=" * 80)
        
        # 创建结果目录
        results_dir = Path("enhanced_multi_model_comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        experiment_start_time = time.time()
        all_results = []
        data_stats = {}
        
        # 加载原始和增强数据
        print(f"\n{'='*80}")
        print(f"📊 数据准备阶段")
        print(f"{'='*80}")
        
        original_texts, original_labels, original_stats = self.load_data(enable_augmentation=False)
        augmented_texts, augmented_labels, augmented_stats = self.load_data(enable_augmentation=True)
        
        data_stats['original'] = original_stats
        data_stats['augmented'] = augmented_stats
        
        # 分析ISCO级别分布
        original_level_analysis = self.analyze_isco_levels(original_labels)
        augmented_level_analysis = self.analyze_isco_levels(augmented_labels)
        
        print(f"\n📈 ISCO级别分布对比:")
        for level in [1, 2, 3, 4]:
            orig_count = original_level_analysis[level]['unique_codes']
            aug_count = augmented_level_analysis[level]['unique_codes']
            print(f"   {level}级({self.isco_levels[level]}): 原始{orig_count} → 增强{aug_count}")
        
        # 对每个模型进行原始vs增强对比
        for model_idx, model_name in enumerate(self.test_models):
            model_display_name = self.model_info[model_name]['name']
            print(f"\n{'='*80}")
            print(f"🤖 模型 {model_idx+1}/{len(self.test_models)}: {model_display_name}")
            print(f"   模型代码: {model_name}")
            print(f"   模型特点: {self.model_info[model_name]['description']}")
            print(f"   参数量: {self.model_info[model_name]['params']}")
            print(f"{'='*80}")
            
            # 原始数据实验
            original_result = self.train_and_evaluate_model(
                model_name, original_texts, original_labels, "Original", results_dir
            )
            all_results.append(original_result)
            
            # 增强数据实验
            augmented_result = self.train_and_evaluate_model(
                model_name, augmented_texts, augmented_labels, "Augmented", results_dir
            )
            all_results.append(augmented_result)
            
            # 打印当前模型的对比结果
            if original_result['status'] == 'success' and augmented_result['status'] == 'success':
                print(f"\n📊 {model_display_name} 对比结果:")
                print(f"   4级准确率: {original_result['accuracy']:.4f} → {augmented_result['accuracy']:.4f} ({(augmented_result['accuracy']-original_result['accuracy'])*100:+.2f}%)")
                for level in [1, 2, 3]:
                    orig_acc = original_result['hierarchical_accuracy'][f'level_{level}_accuracy']
                    aug_acc = augmented_result['hierarchical_accuracy'][f'level_{level}_accuracy']
                    print(f"   {level}级准确率: {orig_acc:.4f} → {aug_acc:.4f} ({(aug_acc-orig_acc)*100:+.2f}%)")
                
                training_time_diff = augmented_result['training_time_minutes'] - original_result['training_time_minutes']
                print(f"   训练时间: {original_result['training_time_minutes']:.1f}分 → {augmented_result['training_time_minutes']:.1f}分 ({training_time_diff:+.1f}分)")
            else:
                print(f"\n⚠️ {model_display_name} 部分实验失败")
                if original_result['status'] != 'success':
                    print(f"   原始数据实验失败: {original_result.get('error', 'Unknown error')}")
                if augmented_result['status'] != 'success':
                    print(f"   增强数据实验失败: {augmented_result.get('error', 'Unknown error')}")
        total_time = time.time() - experiment_start_time
        
        # 生成综合报告
        self.generate_comprehensive_report(all_results, data_stats, total_time, results_dir)
        
        return all_results, results_dir

    def generate_comprehensive_report(self, results, data_stats, total_time, results_dir):
        """生成综合对比报告"""
        print(f"\n📊 生成综合分析报告...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "comprehensive_results.csv", index=False, encoding='utf-8')
        
        # 分组分析结果
        successful_results = [r for r in results if r['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ 没有成功的实验结果")
            return
        
        # 按模型和实验类型分组
        model_comparison = defaultdict(dict)
        for result in successful_results:
            model_name = result['model_name']
            exp_type = result['experiment_name']
            model_comparison[model_name][exp_type] = result
        
        # 计算改进情况
        improvements = []
        level_improvements = {1: [], 2: [], 3: [], 4: []}
        
        for model_name, experiments in model_comparison.items():
            if 'Original' in experiments and 'Augmented' in experiments:
                orig = experiments['Original']
                aug = experiments['Augmented']
                
                # 4级准确率改进
                improvement = aug['accuracy'] - orig['accuracy']
                improvements.append({
                    'model_name': model_name,
                    'original_accuracy': orig['accuracy'],
                    'augmented_accuracy': aug['accuracy'],
                    'improvement': improvement,
                    'improvement_percentage': (improvement / orig['accuracy'] * 100) if orig['accuracy'] > 0 else 0
                })
                
                # 各级别准确率改进
                for level in [1, 2, 3, 4]:
                    orig_acc = orig['hierarchical_accuracy'][f'level_{level}_accuracy']
                    aug_acc = aug['hierarchical_accuracy'][f'level_{level}_accuracy']
                    level_improvements[level].append({
                        'model_name': model_name,
                        'original': orig_acc,
                        'augmented': aug_acc,
                        'improvement': aug_acc - orig_acc
                    })
        
        # 生成详细报告
        report = {
            'experiment_info': {
                'experiment_type': 'Enhanced Multi-Model Data Augmentation Comparison',
                'timestamp': datetime.now().isoformat(),
                'total_time_hours': total_time / 3600,
                'models_tested': self.test_models,
                'max_samples': self.max_samples
            },
            'data_statistics': data_stats,
            'results': results,
            'model_improvements': improvements,
            'level_improvements': level_improvements,
            'summary_statistics': {
                'avg_improvement': np.mean([imp['improvement'] for imp in improvements]) if improvements else 0,
                'best_model': max(improvements, key=lambda x: x['improvement'])['model_name'] if improvements else None,
                'worst_model': min(improvements, key=lambda x: x['improvement'])['model_name'] if improvements else None
            }
        }
        
        # 保存报告
        with open(results_dir / "comprehensive_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成可视化结果表格
        self.create_visualization_tables(model_comparison, results_dir)
        
        # 打印综合结果
        print(f"\n🎉 增强版多模型对比实验完成!")
        print(f"📈 实验总结:")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        print(f"   测试模型: {len(self.test_models)} 个")
        print(f"   成功实验: {len(successful_results)}/{len(results)}")
        
        if improvements:
            print(f"\n🏆 各级别准确率改进排名:")
            
            # 4级准确率排名
            improvements.sort(key=lambda x: x['improvement'], reverse=True)
            print(f"\n   4级准确率改进:")
            for i, imp in enumerate(improvements, 1):
                model_short = imp['model_name'].split('/')[-1]
                print(f"     {i}. {model_short}: {imp['improvement']:.4f} ({imp['improvement_percentage']:+.2f}%)")
            
            # 各级别改进排名
            for level in [1, 2, 3]:
                level_imps = level_improvements[level]
                level_imps.sort(key=lambda x: x['improvement'], reverse=True)
                print(f"\n   {level}级准确率改进:")
                for i, imp in enumerate(level_imps, 1):
                    model_short = imp['model_name'].split('/')[-1]
                    improvement_pct = (imp['improvement'] / imp['original'] * 100) if imp['original'] > 0 else 0
                    print(f"     {i}. {model_short}: {imp['improvement']:.4f} ({improvement_pct:+.2f}%)")
            
            # 最佳模型推荐
            best_model = report['summary_statistics']['best_model']
            avg_improvement = report['summary_statistics']['avg_improvement']
            print(f"\n🎯 实验结论:")
            print(f"   最佳改进模型: {best_model.split('/')[-1]}")
            print(f"   平均准确率提升: {avg_improvement:.4f} ({avg_improvement*100:+.2f}%)")
            
            if avg_improvement > 0.01:
                print(f"   🎉 数据增强效果显著！")
            elif avg_improvement > 0:
                print(f"   ⚖️ 数据增强效果中性")
            else:
                print(f"   ⚠️ 数据增强需要优化")
        
        print(f"\n📁 详细结果保存在: {results_dir}")
        
        # 后续分析建议
        print(f"\n💡 深度分析建议:")
        print(f"   1. 查看 comprehensive_results.csv 了解所有模型详细指标")
        print(f"   2. 查看 model_comparison_table.csv 了解模型对比")
        print(f"   3. 查看 level_accuracy_comparison.csv 了解各级别准确率")
        print(f"   4. 使用 create_visualizations() 生成图表")

    def create_visualization_tables(self, model_comparison, results_dir):
        """创建可视化对比表格"""
        
        # 模型对比表
        comparison_data = []
        for model_name, experiments in model_comparison.items():
            if 'Original' in experiments and 'Augmented' in experiments:
                orig = experiments['Original']
                aug = experiments['Augmented']
                
                row = {
                    'Model': orig.get('model_display_name', model_name.split('/')[-1]),
                    'Model_Code': model_name,
                    'Description': orig.get('model_description', ''),
                    'Original_4Level_Acc': orig['accuracy'],
                    'Augmented_4Level_Acc': aug['accuracy'],
                    '4Level_Improvement': aug['accuracy'] - orig['accuracy'],
                    '4Level_Improvement_Pct': ((aug['accuracy'] - orig['accuracy']) / orig['accuracy'] * 100) if orig['accuracy'] > 0 else 0,
                    'Original_3Level_Acc': orig['hierarchical_accuracy']['level_3_accuracy'],
                    'Augmented_3Level_Acc': aug['hierarchical_accuracy']['level_3_accuracy'],
                    '3Level_Improvement': aug['hierarchical_accuracy']['level_3_accuracy'] - orig['hierarchical_accuracy']['level_3_accuracy'],
                    'Original_2Level_Acc': orig['hierarchical_accuracy']['level_2_accuracy'],
                    'Augmented_2Level_Acc': aug['hierarchical_accuracy']['level_2_accuracy'],
                    '2Level_Improvement': aug['hierarchical_accuracy']['level_2_accuracy'] - orig['hierarchical_accuracy']['level_2_accuracy'],
                    'Original_1Level_Acc': orig['hierarchical_accuracy']['level_1_accuracy'],
                    'Augmented_1Level_Acc': aug['hierarchical_accuracy']['level_1_accuracy'],
                    '1Level_Improvement': aug['hierarchical_accuracy']['level_1_accuracy'] - orig['hierarchical_accuracy']['level_1_accuracy'],
                    'Training_Time_Increase': aug['training_time_minutes'] - orig['training_time_minutes'],
                    'Original_LR': orig.get('learning_rate', 'N/A'),
                    'Augmented_LR': aug.get('learning_rate', 'N/A')
                }
                comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(results_dir / "model_comparison_table.csv", index=False)
        
        # 级别准确率对比表
        level_data = []
        for model_name, experiments in model_comparison.items():
            if 'Original' in experiments and 'Augmented' in experiments:
                orig = experiments['Original']
                aug = experiments['Augmented']
                
                for level in [1, 2, 3, 4]:
                    level_data.append({
                        'Model': orig.get('model_display_name', model_name.split('/')[-1]),
                        'Model_Code': model_name,
                        'Level': level,
                        'Level_Name': self.isco_levels[level],
                        'Original_Accuracy': orig['hierarchical_accuracy'][f'level_{level}_accuracy'],
                        'Augmented_Accuracy': aug['hierarchical_accuracy'][f'level_{level}_accuracy'],
                        'Improvement': aug['hierarchical_accuracy'][f'level_{level}_accuracy'] - orig['hierarchical_accuracy'][f'level_{level}_accuracy'],
                        'Improvement_Percentage': ((aug['hierarchical_accuracy'][f'level_{level}_accuracy'] - orig['hierarchical_accuracy'][f'level_{level}_accuracy']) / orig['hierarchical_accuracy'][f'level_{level}_accuracy'] * 100) if orig['hierarchical_accuracy'][f'level_{level}_accuracy'] > 0 else 0
                    })
        
        level_df = pd.DataFrame(level_data)
        level_df.to_csv(results_dir / "level_accuracy_comparison.csv", index=False)
        
        print(f"✅ 可视化表格已生成:")
        print(f"   - model_comparison_table.csv: 模型横向对比")
        print(f"   - level_accuracy_comparison.csv: 级别准确率详细对比")

    def create_visualizations(self, results_dir):
        """创建可视化图表"""
        try:
            # 读取对比数据
            comparison_df = pd.read_csv(results_dir / "model_comparison_table.csv")
            level_df = pd.read_csv(results_dir / "level_accuracy_comparison.csv")
            
            # 设置matplotlib中文支持和更好的字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['figure.facecolor'] = 'white'
            plt.rcParams['axes.facecolor'] = 'white'
            
            # 1. 模型准确率对比图 - 更紧凑的布局
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            colors_original = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
            colors_augmented = ['#2980b9', '#c0392b', '#27ae60', '#d68910', '#8e44ad']
            
            for i, level in enumerate([4, 3, 2, 1]):
                ax = axes[i//2, i%2]
                
                models = comparison_df['Model']
                original_acc = comparison_df[f'Original_{level}Level_Acc']
                augmented_acc = comparison_df[f'Augmented_{level}Level_Acc']
                
                x = np.arange(len(models))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, original_acc, width, label='原始数据', 
                              color=colors_original[:len(models)], alpha=0.8, edgecolor='white', linewidth=1)
                bars2 = ax.bar(x + width/2, augmented_acc, width, label='数据增强', 
                              color=colors_augmented[:len(models)], alpha=0.8, edgecolor='white', linewidth=1)
                
                ax.set_ylabel('准确率', fontsize=12, fontweight='bold')
                ax.set_title(f'{level}级准确率对比\n{self.isco_levels[level]}', fontsize=14, fontweight='bold', pad=20)
                ax.set_xticks(x)
                
                # 缩短模型名称显示
                short_names = []
                for model in models:
                    if 'Google' in model:
                        short_names.append('Google BERT')
                    elif 'HFL' in model and 'RoBERTa' in model:
                        short_names.append('HFL RoBERTa')
                    elif 'HFL' in model and 'BERT' in model:
                        short_names.append('HFL BERT')
                    elif 'CKIP' in model:
                        short_names.append('CKIP BERT')
                    elif 'Multilingual' in model:
                        short_names.append('Multilingual')
                    else:
                        short_names.append(model)
                
                ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=10, loc='upper left')
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, max(max(original_acc), max(augmented_acc)) * 1.1)
                
                # 添加数值标签
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # 添加改进幅度标注
                for j, (orig, aug) in enumerate(zip(original_acc, augmented_acc)):
                    improvement = aug - orig
                    if improvement > 0:
                        ax.annotate(f'+{improvement:.3f}', 
                                  xy=(j, max(orig, aug) + 0.02), 
                                  ha='center', va='bottom', 
                                  fontsize=8, color='green', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
                    elif improvement < 0:
                        ax.annotate(f'{improvement:.3f}', 
                                  xy=(j, max(orig, aug) + 0.02), 
                                  ha='center', va='bottom', 
                                  fontsize=8, color='red', fontweight='bold',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.7))
            
            plt.suptitle('各级别准确率对比 - 原始数据 vs 数据增强', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(results_dir / "model_accuracy_comparison.png", dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # 2. 改进幅度热力图 - 增强版
            fig, ax = plt.subplots(figsize=(12, 8))
            
            improvement_matrix = []
            model_display_names = []
            level_names = ['1级\n主要职业组', '2级\n次要职业组', '3级\n次级职业组', '4级\n基本职业组']
            
            for _, row in comparison_df.iterrows():
                model_display_names.append(row['Model'])
                improvements = [
                    row['1Level_Improvement'],
                    row['2Level_Improvement'], 
                    row['3Level_Improvement'],
                    row['4Level_Improvement']
                ]
                improvement_matrix.append(improvements)
            
            improvement_matrix = np.array(improvement_matrix)
            
            # 使用更好的配色方案
            im = ax.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.02, vmax=0.05)
            
            # 设置标签
            ax.set_xticks(np.arange(len(level_names)))
            ax.set_yticks(np.arange(len(model_display_names)))
            ax.set_xticklabels(level_names, fontsize=12, fontweight='bold')
            ax.set_yticklabels(model_display_names, fontsize=12, fontweight='bold')
            
            # 添加数值和百分比
            for i in range(len(model_display_names)):
                for j in range(len(level_names)):
                    improvement = improvement_matrix[i, j]
                    percentage = improvement * 100
                    text_color = 'white' if abs(improvement) > 0.02 else 'black'
                    ax.text(j, i, f'{improvement:.3f}\n({percentage:+.1f}%)',
                           ha="center", va="center", color=text_color, 
                           fontweight='bold', fontsize=10)
            
            ax.set_title("数据增强改进效果热力图", fontsize=16, fontweight='bold', pad=20)
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('准确率改进', fontsize=12, fontweight='bold')
            cbar.ax.tick_params(labelsize=10)
            
            plt.tight_layout()
            plt.savefig(results_dir / "improvement_heatmap.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # 3. 级别准确率趋势图 - 改进版
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
            
            # 左图：原始数据趋势
            for i, model in enumerate(comparison_df['Model'].unique()):
                model_data = level_df[level_df['Model'] == model]
                levels = model_data['Level']
                original_acc = model_data['Original_Accuracy']
                
                ax1.plot(levels, original_acc, 'o-', label=model, linewidth=2.5, 
                        markersize=8, color=colors_original[i], alpha=0.8)
            
            ax1.set_xlabel('ISCO 级别', fontsize=12, fontweight='bold')
            ax1.set_ylabel('准确率', fontsize=12, fontweight='bold')
            ax1.set_title('原始数据 - 各级别准确率趋势', fontsize=14, fontweight='bold')
            ax1.set_xticks([1, 2, 3, 4])
            ax1.set_xticklabels(['1级\n主要职业组', '2级\n次要职业组', '3级\n次级职业组', '4级\n基本职业组'])
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 右图：数据增强趋势
            for i, model in enumerate(comparison_df['Model'].unique()):
                model_data = level_df[level_df['Model'] == model]
                levels = model_data['Level']
                augmented_acc = model_data['Augmented_Accuracy']
                
                ax2.plot(levels, augmented_acc, 's-', label=model, linewidth=2.5, 
                        markersize=8, color=colors_augmented[i], alpha=0.8)
            
            ax2.set_xlabel('ISCO 级别', fontsize=12, fontweight='bold')
            ax2.set_ylabel('准确率', fontsize=12, fontweight='bold')
            ax2.set_title('数据增强 - 各级别准确率趋势', fontsize=14, fontweight='bold')
            ax2.set_xticks([1, 2, 3, 4])
            ax2.set_xticklabels(['1级\n主要职业组', '2级\n次要职业组', '3级\n次级职业组', '4级\n基本职业组'])
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle('各级别准确率趋势对比', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)
            plt.savefig(results_dir / "level_accuracy_trends.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            # 4. 新增：模型排名图
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 计算平均改进
            avg_improvements = []
            model_names = []
            for _, row in comparison_df.iterrows():
                avg_imp = (row['1Level_Improvement'] + row['2Level_Improvement'] + 
                          row['3Level_Improvement'] + row['4Level_Improvement']) / 4
                avg_improvements.append(avg_imp)
                model_names.append(row['Model'])
            
            # 排序
            sorted_data = sorted(zip(model_names, avg_improvements), key=lambda x: x[1], reverse=True)
            sorted_models, sorted_improvements = zip(*sorted_data)
            
            colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in sorted_improvements]
            bars = ax.barh(range(len(sorted_models)), sorted_improvements, color=colors, alpha=0.8)
            
            ax.set_yticks(range(len(sorted_models)))
            ax.set_yticklabels(sorted_models, fontsize=12, fontweight='bold')
            ax.set_xlabel('平均准确率改进', fontsize=12, fontweight='bold')
            ax.set_title('模型数据增强效果排名', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, axis='x')
            
            # 添加数值标签
            for i, (bar, imp) in enumerate(zip(bars, sorted_improvements)):
                width = bar.get_width()
                ax.text(width + (0.001 if width >= 0 else -0.001), bar.get_y() + bar.get_height()/2,
                       f'{imp:.4f} ({imp*100:+.2f}%)', 
                       ha='left' if width >= 0 else 'right', va='center', 
                       fontweight='bold', fontsize=11)
            
            plt.tight_layout()
            plt.savefig(results_dir / "model_ranking.png", dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"✅ 可视化图表已生成:")
            print(f"   - model_accuracy_comparison.png: 各级别准确率对比 (含改进标注)")
            print(f"   - improvement_heatmap.png: 改进效果热力图 (含百分比)")
            print(f"   - level_accuracy_trends.png: 级别准确率趋势对比")
            print(f"   - model_ranking.png: 模型效果排名图")
            print(f"   图表特点: 中文支持、颜色区分、数值标注、专业布局")
            
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
            print("请检查是否安装了matplotlib: pip install matplotlib")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    print("🔬 增强版多模型数据增强效果对比实验")
    print("支持多级别准确率分析和全面模型对比")
    print("=" * 80)
    
    # 配置实验
    csv_path = "lunwenimpro/newjob1_sortall.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        print("请确保数据文件在当前目录下")
        return
    
    print("选择实验规模:")
    print("1. 快速测试 (8K样本, 2个模型)")
    print("2. 标准对比 (12K样本, 全部模型)")
    print("3. 完整对比 (无限制, 全部模型)")
    print("4. 自定义配置")
    
    choice = input("请输入选择 (1-4): ")
    
    if choice == "1":
        max_samples = 8000
        test_models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
    elif choice == "2":
        max_samples = 12000
        test_models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
    elif choice == "3":
        max_samples = None
        test_models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
    elif choice == "4":
        max_samples = int(input("最大样本数 (留空表示无限制): ") or 0) or None
        print("选择测试模型 (空格分隔序号，默认选择所有):")
        models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
        model_names = [
            'Google Chinese BERT',
            'HFL Chinese BERT-wwm',
            'HFL Chinese RoBERTa', 
            'CKIP Chinese BERT',
            'BERT Multilingual'
        ]
        
        for i, (model, name) in enumerate(zip(models, model_names), 1):
            print(f"  {i}. {name} ({model})")
        
        selected = input("输入模型序号 (默认全选): ").split()
        if selected:
            test_models = [models[int(i)-1] for i in selected if i.isdigit() and 1 <= int(i) <= len(models)]
        else:
            test_models = models  # 默认全选
        
        if not test_models:
            test_models = ['bert-base-chinese']
    else:
        print("无效选择，使用快速测试")
        max_samples = 8000
        test_models = [
            'bert-base-chinese',                    # Google Chinese BERT
            'hfl/chinese-bert-wwm-ext',            # HFL Chinese BERT-wwm  
            'hfl/chinese-roberta-wwm-ext',         # HFL Chinese RoBERTa
            'ckiplab/bert-base-chinese',           # CKIP Chinese BERT
            'bert-base-multilingual-cased'        # BERT Multilingual
        ]
    
    print(f"\n🎯 实验配置:")
    print(f"   数据文件: {csv_path}")
    print(f"   样本限制: {max_samples if max_samples else '无限制'}")
    print(f"   测试模型: {len(test_models)} 个")
    for i, model in enumerate(test_models, 1):
        print(f"     {i}. {model}")
    print(f"   对比维度: 原始数据 vs 数据增强")
    print(f"   分析级别: ISCO 1-4级层次准确率")
    
    confirm = input(f"\n确认开始多模型对比实验? (y/N): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return
    
    try:
        # 创建实验对象
        experiment = EnhancedMultiModelComparison(csv_path, max_samples)
        experiment.test_models = test_models
        
        # 运行对比实验
        results, results_dir = experiment.run_comprehensive_comparison()
        
        # 生成可视化图表
        print(f"\n📊 生成可视化图表...")
        experiment.create_visualizations(results_dir)
        
        print(f"\n🎯 多模型对比实验完成!")
        print(f"📁 查看详细结果: {results_dir}")
        print(f"🔍 主要文件:")
        print(f"   - comprehensive_report.json: 完整实验报告")
        print(f"   - model_comparison_table.csv: 模型对比表格")
        print(f"   - level_accuracy_comparison.csv: 级别准确率详情")
        print(f"   - *.png: 可视化图表")
        
    except Exception as e:
        print(f"❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()