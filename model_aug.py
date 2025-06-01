#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面多模型对比分析系统
支持多个模型、多个Top-K级别、增强前后详细对比
"""

import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import torch
torch.set_float32_matmul_precision('medium')

# 设置HuggingFace镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 导入原有模块
if __name__ == '__main__':
    from job_offers_classifier.job_offers_classfier import (
        ChineseTransformerJobOffersClassifier,
        get_recommended_chinese_models
    )
    from job_offers_classifier.job_offers_utils import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
    
    # 导入数据增强模块
    from chinese_job_data_augmentation import EnhancedJobDataProcessor


class ComprehensiveModelComparison:
    """全面的多模型对比分析系统"""
    
    def __init__(self, csv_path: str, max_samples: int = 10000):
        self.csv_path = csv_path
        self.max_samples = max_samples
        
        # 多个测试模型配置
        self.test_models = {
            'google_bert': {
                'name': 'bert-base-chinese',
                'description': 'Google Chinese BERT',
                'batch_size': 16,
                'learning_rate': 2e-5
            },
            'hfl_bert': {
                'name': 'hfl/chinese-bert-wwm-ext',
                'description': 'HFL Chinese BERT-wwm-ext',
                'batch_size': 16,
                'learning_rate': 2e-5
            },
            'hfl_roberta': {
                'name': 'hfl/chinese-roberta-wwm-ext',
                'description': 'HFL Chinese RoBERTa-wwm-ext',
                'batch_size': 16,
                'learning_rate': 2e-5
            }
        }
        
        # 训练配置
        self.training_config = {
            'max_epochs': 4,
            'patience': 2,
            'max_seq_length': 256
        }
        
        # Top-K评估级别
        self.top_k_levels = [1, 3, 5, 10]
        
        # 结果存储
        self.all_results = []
        
        print("🎯 全面多模型对比分析系统初始化")
        print(f"   将测试 {len(self.test_models)} 个模型")
        print(f"   Top-K评估级别: {self.top_k_levels}")

    def test_model_availability(self, model_name):
        """测试单个模型是否可用"""
        try:
            from transformers import AutoConfig, AutoTokenizer
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return True
        except Exception as e:
            print(f"   ❌ {model_name} 不可用: {str(e)[:100]}...")
            return False

    def filter_available_models(self):
        """筛选可用的模型"""
        print("🧪 检测模型可用性...")
        available_models = {}
        
        for key, config in self.test_models.items():
            model_name = config['name']
            print(f"   测试 {model_name}...")
            
            if self.test_model_availability(model_name):
                available_models[key] = config
                print(f"   ✅ {config['description']} 可用")
            else:
                print(f"   ❌ {config['description']} 不可用")
        
        self.test_models = available_models
        print(f"✅ 找到 {len(available_models)} 个可用模型")
        return len(available_models) > 0

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

    def safe_data_split(self, texts, labels):
        """安全的数据划分"""
        from collections import Counter
        
        label_counts = Counter(labels)
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        print(f"   单样本类别: {len(single_sample_classes)}")
        print(f"   多样本类别: {len(multi_sample_classes)}")
        
        train_indices = []
        test_indices = []
        
        # 单样本类别全部放入训练集
        for i, (text, label) in enumerate(zip(texts, labels)):
            if label in single_sample_classes:
                train_indices.append(i)
        
        # 多样本类别分层划分
        if multi_sample_classes:
            multi_data = [(i, text, label) for i, (text, label) in enumerate(zip(texts, labels)) 
                         if label in multi_sample_classes]
            
            if len(multi_data) > 0:
                multi_indices = [item[0] for item in multi_data]
                multi_labels = [item[2] for item in multi_data]
                
                try:
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_indices)), 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=multi_labels
                    )
                    
                    train_indices.extend([multi_indices[i] for i in multi_train_idx])
                    test_indices.extend([multi_indices[i] for i in multi_test_idx])
                except ValueError:
                    # 分层失败，使用随机划分
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_indices)), 
                        test_size=0.2, 
                        random_state=42
                    )
                    train_indices.extend([multi_indices[i] for i in multi_train_idx])
                    test_indices.extend([multi_indices[i] for i in multi_test_idx])
        
        # 构建最终数据集
        train_texts = [texts[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_texts = [texts[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        return train_texts, train_labels, test_texts, test_labels

    def calculate_comprehensive_metrics(self, y_true, predictions_df, experiment_info):
        """计算全面的评估指标"""
        metrics = {
            'experiment_info': experiment_info,
            'basic_metrics': {},
            'top_k_metrics': {},
            'class_level_metrics': {},
            'hierarchy_metrics': {}
        }
        
        # 基础指标
        y_pred = predictions_df['class_1'].tolist()
        metrics['basic_metrics']['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 计算各个Top-K准确率
        for k in self.top_k_levels:
            if k <= len(predictions_df.columns) // 2:  # 确保有足够的预测列
                top_k_acc = sum(
                    true_label in [predictions_df.iloc[i][f'class_{j}'] for j in range(1, min(k+1, 6))]
                    for i, true_label in enumerate(y_true)
                ) / len(y_true)
                metrics['top_k_metrics'][f'top_{k}_accuracy'] = top_k_acc
        
        # 置信度分析
        confidences = predictions_df['prob_1'].tolist()
        correct_mask = [t == p for t, p in zip(y_true, y_pred)]
        
        metrics['basic_metrics']['avg_confidence'] = np.mean(confidences)
        metrics['basic_metrics']['avg_confidence_correct'] = np.mean([conf for conf, correct in zip(confidences, correct_mask) if correct]) if any(correct_mask) else 0
        metrics['basic_metrics']['avg_confidence_wrong'] = np.mean([conf for conf, correct in zip(confidences, correct_mask) if not correct]) if not all(correct_mask) else 0
        
        # 类别级别分析
        from collections import defaultdict
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidences': []})
        
        for true_label, pred_label, confidence in zip(y_true, y_pred, confidences):
            class_stats[true_label]['total'] += 1
            class_stats[true_label]['confidences'].append(confidence)
            if true_label == pred_label:
                class_stats[true_label]['correct'] += 1
        
        # 计算每个类别的准确率
        class_accuracies = {}
        for class_label, stats in class_stats.items():
            if stats['total'] > 0:
                class_accuracies[class_label] = {
                    'accuracy': stats['correct'] / stats['total'],
                    'sample_count': stats['total'],
                    'avg_confidence': np.mean(stats['confidences'])
                }
        
        metrics['class_level_metrics'] = class_accuracies
        
        # ISCO层次化分析
        hierarchy_stats = self.analyze_isco_hierarchy_performance(y_true, y_pred)
        metrics['hierarchy_metrics'] = hierarchy_stats
        
        return metrics

    def analyze_isco_hierarchy_performance(self, y_true, y_pred):
        """分析ISCO层次结构的性能"""
        hierarchy_stats = {}
        
        # 按ISCO层级分析
        for level in [1, 2, 3, 4]:
            level_true = [label[:level] for label in y_true]
            level_pred = [label[:level] for label in y_pred]
            
            level_accuracy = accuracy_score(level_true, level_pred)
            hierarchy_stats[f'level_{level}_accuracy'] = level_accuracy
            
            # 计算该层级的类别数
            unique_true = len(set(level_true))
            unique_pred = len(set(level_pred))
            hierarchy_stats[f'level_{level}_true_classes'] = unique_true
            hierarchy_stats[f'level_{level}_pred_classes'] = unique_pred
        
        return hierarchy_stats

    def train_single_model_experiment(self, model_key, model_config, train_texts, train_labels, 
                                    test_texts, test_labels, hierarchy, results_dir, 
                                    experiment_name, is_augmented=False):
        """训练单个模型实验"""
        print(f"\n🤖 开始训练: {experiment_name}")
        print(f"   模型: {model_config['description']}")
        print(f"   数据类型: {'增强数据' if is_augmented else '原始数据'}")
        
        start_time = time.time()
        model_dir = results_dir / f"model_{model_key}_{experiment_name.lower().replace(' ', '_')}"
        
        try:
            # 创建分类器
            classifier = ChineseTransformerJobOffersClassifier(
                model_dir=str(model_dir),
                hierarchy=hierarchy,
                transformer_model=model_config['name'],
                learning_rate=model_config['learning_rate'],
                batch_size=model_config['batch_size'],
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
            val_texts = test_texts[:val_size]
            val_labels = test_labels[:val_size]
            final_test_texts = test_texts[val_size:]
            final_test_labels = test_labels[val_size:]
            
            print(f"   训练样本: {len(train_texts):,}")
            print(f"   验证样本: {val_size:,}")
            print(f"   测试样本: {len(final_test_texts):,}")
            
            # 训练
            print(f"   🎯 开始训练...")
            classifier.fit(train_labels, train_texts, y_val=val_labels, X_val=val_texts)
            
            # 预测
            print(f"   🔮 预测中...")
            max_top_k = max(self.top_k_levels)
            predictions_df = classifier.predict(final_test_texts, format='dataframe', top_k=max_top_k)
            
            training_time = time.time() - start_time
            
            # 计算全面指标
            experiment_info = {
                'model_key': model_key,
                'model_name': model_config['name'],
                'model_description': model_config['description'],
                'experiment_name': experiment_name,
                'is_augmented': is_augmented,
                'train_samples': len(train_texts),
                'test_samples': len(final_test_texts),
                'training_time_minutes': training_time / 60
            }
            
            metrics = self.calculate_comprehensive_metrics(final_test_labels, predictions_df, experiment_info)
            metrics['status'] = 'success'
            
            # 显示结果
            print(f"✅ {experiment_name} 训练完成!")
            print(f"   基础准确率: {metrics['basic_metrics']['accuracy']:.4f}")
            
            for k in self.top_k_levels:
                if f'top_{k}_accuracy' in metrics['top_k_metrics']:
                    acc = metrics['top_k_metrics'][f'top_{k}_accuracy']
                    print(f"   Top-{k}准确率: {acc:.4f}")
            
            print(f"   训练时间: {training_time/60:.1f} 分钟")
            
            # 保存详细结果
            detailed_results = pd.DataFrame({
                'true_label': final_test_labels,
                'predicted_label': predictions_df['class_1'].tolist(),
                'confidence': predictions_df['prob_1'].tolist(),
                'correct': [t == p for t, p in zip(final_test_labels, predictions_df['class_1'].tolist())]
            })
            
            # 添加Top-K预测
            for k in range(2, min(max_top_k + 1, 6)):
                if f'class_{k}' in predictions_df.columns:
                    detailed_results[f'top_{k}_pred'] = predictions_df[f'class_{k}'].tolist()
                    detailed_results[f'top_{k}_prob'] = predictions_df[f'prob_{k}'].tolist()
            
            detailed_results.to_csv(
                results_dir / f"detailed_{model_key}_{experiment_name.lower().replace(' ', '_')}.csv", 
                index=False, encoding='utf-8'
            )
            
            return metrics
            
        except Exception as e:
            print(f"❌ {experiment_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'experiment_info': {
                    'model_key': model_key,
                    'model_name': model_config['name'],
                    'model_description': model_config['description'],
                    'experiment_name': experiment_name,
                    'is_augmented': is_augmented
                },
                'status': 'failed',
                'error': str(e)
            }

    def load_data(self, use_augmentation=False):
        """加载数据"""
        if use_augmentation:
            print(f"\n🎯 加载增强数据...")
            processor = EnhancedJobDataProcessor()
            texts, labels, processing_stats = processor.process_csv_data(
                csv_path=self.csv_path,
                enable_augmentation=True,
                balance_data=True,
                target_samples_per_class=6
            )
            
            if self.max_samples and len(texts) > self.max_samples:
                # 智能采样
                from collections import Counter
                label_counts = Counter(labels)
                
                # 保留单样本类别，对多样本类别采样
                single_classes = [label for label, count in label_counts.items() if count == 1]
                multi_classes = [label for label, count in label_counts.items() if count > 1]
                
                sampled_texts = []
                sampled_labels = []
                
                # 保留所有单样本类别
                for text, label in zip(texts, labels):
                    if label in single_classes:
                        sampled_texts.append(text)
                        sampled_labels.append(label)
                
                # 从多样本类别中采样
                remaining_budget = self.max_samples - len(sampled_texts)
                if remaining_budget > 0:
                    multi_data = [(text, label) for text, label in zip(texts, labels) if label in multi_classes]
                    if len(multi_data) > remaining_budget:
                        indices = np.random.choice(len(multi_data), size=remaining_budget, replace=False)
                        for idx in indices:
                            sampled_texts.append(multi_data[idx][0])
                            sampled_labels.append(multi_data[idx][1])
                    else:
                        for text, label in multi_data:
                            sampled_texts.append(text)
                            sampled_labels.append(label)
                
                texts, labels = sampled_texts, sampled_labels
                print(f"   智能采样后: {len(texts)} 样本")
            
        else:
            print(f"\n📊 加载原始数据...")
            # 简化的原始数据加载
            try:
                df = pd.read_csv(self.csv_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.csv_path, encoding='gbk')
            
            # 基础文本处理
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
            df = df[df['combined_text'].str.strip() != ''].copy()
            
            # 智能采样原始数据
            if self.max_samples and len(df) > self.max_samples:
                class_counts = df['isco_code'].value_counts()
                sufficient_classes = class_counts[class_counts >= 2].index.tolist()
                
                if len(sufficient_classes) > 0:
                    sufficient_df = df[df['isco_code'].isin(sufficient_classes)]
                    if len(sufficient_df) > self.max_samples:
                        sampled_df = sufficient_df.groupby('isco_code', group_keys=False).apply(
                            lambda x: x.sample(min(len(x), max(2, self.max_samples // len(sufficient_classes))), 
                                             random_state=42)
                        ).reset_index(drop=True)
                        
                        if len(sampled_df) > self.max_samples:
                            sampled_df = sampled_df.sample(n=self.max_samples, random_state=42)
                    else:
                        sampled_df = sufficient_df
                    
                    df = sampled_df
            
            texts = df['combined_text'].tolist()
            labels = df['isco_code'].tolist()
        
        print(f"✅ 数据加载完成: {len(texts)} 样本, {len(set(labels))} 类别")
        return texts, labels

    def run_comprehensive_comparison(self):
        """运行全面对比实验"""
        print("🎯 开始全面多模型对比分析")
        print("=" * 70)
        
        # 检查模型可用性
        if not self.filter_available_models():
            print("❌ 没有可用的模型")
            return None, None
        
        # 创建结果目录
        results_dir = Path("comprehensive_model_comparison")
        results_dir.mkdir(exist_ok=True)
        
        total_start_time = time.time()
        
        # 对每个模型进行原始数据和增强数据的对比
        for model_key, model_config in self.test_models.items():
            print(f"\n{'='*70}")
            print(f"🔬 测试模型: {model_config['description']}")
            print(f"{'='*70}")
            
            # 加载原始数据
            original_texts, original_labels = self.load_data(use_augmentation=False)
            train_texts_orig, train_labels_orig, test_texts_orig, test_labels_orig = self.safe_data_split(original_texts, original_labels)
            
            # 加载增强数据
            augmented_texts, augmented_labels = self.load_data(use_augmentation=True)
            train_texts_aug, train_labels_aug, test_texts_aug, test_labels_aug = self.safe_data_split(augmented_texts, augmented_labels)
            
            # 创建层次结构
            hierarchy = self.create_isco_hierarchy_from_codes(set(original_labels + augmented_labels))
            
            # 训练原始数据模型
            original_result = self.train_single_model_experiment(
                model_key, model_config, 
                train_texts_orig, train_labels_orig, 
                test_texts_orig, test_labels_orig, 
                hierarchy, results_dir, 
                f"{model_key}_original", is_augmented=False
            )
            self.all_results.append(original_result)
            
            # 训练增强数据模型
            augmented_result = self.train_single_model_experiment(
                model_key, model_config, 
                train_texts_aug, train_labels_aug, 
                test_texts_aug, test_labels_aug, 
                hierarchy, results_dir, 
                f"{model_key}_augmented", is_augmented=True
            )
            self.all_results.append(augmented_result)
        
        total_time = time.time() - total_start_time
        
        # 生成综合分析报告
        self.generate_comprehensive_report(total_time, results_dir)
        
        return self.all_results, results_dir

    def generate_comprehensive_report(self, total_time, results_dir):
        """生成综合分析报告"""
        print(f"\n📊 生成综合分析报告...")
        
        # 创建结果对比表
        comparison_data = []
        improvements = {}
        
        for result in self.all_results:
            if result['status'] == 'success':
                info = result['experiment_info']
                
                row = {
                    'model': info['model_description'],
                    'data_type': '增强数据' if info['is_augmented'] else '原始数据',
                    'train_samples': info['train_samples'],
                    'test_samples': info['test_samples'],
                    'training_time_min': info['training_time_minutes'],
                    'accuracy': result['basic_metrics']['accuracy'],
                    'avg_confidence': result['basic_metrics']['avg_confidence']
                }
                
                # 添加Top-K准确率
                for k in self.top_k_levels:
                    if f'top_{k}_accuracy' in result['top_k_metrics']:
                        row[f'top_{k}_acc'] = result['top_k_metrics'][f'top_{k}_accuracy']
                
                # 添加层次准确率
                for level in [1, 2, 3, 4]:
                    if f'level_{level}_accuracy' in result['hierarchy_metrics']:
                        row[f'isco_{level}_acc'] = result['hierarchy_metrics'][f'level_{level}_accuracy']
                
                comparison_data.append(row)
                
                # 计算改进情况
                model_key = info['model_key']
                if model_key not in improvements:
                    improvements[model_key] = {'original': None, 'augmented': None}
                
                if info['is_augmented']:
                    improvements[model_key]['augmented'] = result
                else:
                    improvements[model_key]['original'] = result
        
        # 保存对比表
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(results_dir / "comprehensive_comparison.csv", index=False, encoding='utf-8')
        
        # 计算改进统计
        improvement_stats = []
        
        for model_key, data in improvements.items():
            if data['original'] and data['augmented'] and data['original']['status'] == 'success' and data['augmented']['status'] == 'success':
                orig = data['original']
                aug = data['augmented']
                
                model_name = orig['experiment_info']['model_description']
                
                # 基础指标改进
                acc_improvement = aug['basic_metrics']['accuracy'] - orig['basic_metrics']['accuracy']
                acc_improvement_pct = (acc_improvement / orig['basic_metrics']['accuracy'] * 100) if orig['basic_metrics']['accuracy'] > 0 else 0
                
                improvement_row = {
                    'model': model_name,
                    'original_accuracy': orig['basic_metrics']['accuracy'],
                    'augmented_accuracy': aug['basic_metrics']['accuracy'],
                    'accuracy_improvement': acc_improvement,
                    'accuracy_improvement_pct': acc_improvement_pct,
                    'original_samples': orig['experiment_info']['train_samples'],
                    'augmented_samples': aug['experiment_info']['train_samples'],
                    'sample_increase_ratio': aug['experiment_info']['train_samples'] / orig['experiment_info']['train_samples'],
                    'training_time_increase_min': aug['experiment_info']['training_time_minutes'] - orig['experiment_info']['training_time_minutes']
                }
                
                # Top-K改进
                for k in self.top_k_levels:
                    if f'top_{k}_accuracy' in orig['top_k_metrics'] and f'top_{k}_accuracy' in aug['top_k_metrics']:
                        orig_acc = orig['top_k_metrics'][f'top_{k}_accuracy']
                        aug_acc = aug['top_k_metrics'][f'top_{k}_accuracy']
                        improvement = aug_acc - orig_acc
                        improvement_pct = (improvement / orig_acc * 100) if orig_acc > 0 else 0
                        
                        improvement_row[f'top_{k}_original'] = orig_acc
                        improvement_row[f'top_{k}_augmented'] = aug_acc
                        improvement_row[f'top_{k}_improvement'] = improvement
                        improvement_row[f'top_{k}_improvement_pct'] = improvement_pct
                
                # ISCO层次改进
                for level in [1, 2, 3, 4]:
                    if f'level_{level}_accuracy' in orig['hierarchy_metrics'] and f'level_{level}_accuracy' in aug['hierarchy_metrics']:
                        orig_acc = orig['hierarchy_metrics'][f'level_{level}_accuracy']
                        aug_acc = aug['hierarchy_metrics'][f'level_{level}_accuracy']
                        improvement = aug_acc - orig_acc
                        improvement_pct = (improvement / orig_acc * 100) if orig_acc > 0 else 0
                        
                        improvement_row[f'isco_{level}_original'] = orig_acc
                        improvement_row[f'isco_{level}_augmented'] = aug_acc
                        improvement_row[f'isco_{level}_improvement'] = improvement
                        improvement_row[f'isco_{level}_improvement_pct'] = improvement_pct
                
                improvement_stats.append(improvement_row)
        
        # 保存改进统计
        if improvement_stats:
            improvement_df = pd.DataFrame(improvement_stats)
            improvement_df.to_csv(results_dir / "improvement_analysis.csv", index=False, encoding='utf-8')
        
        # 生成文本报告
        self.generate_text_report(comparison_df, improvement_stats, total_time, results_dir)