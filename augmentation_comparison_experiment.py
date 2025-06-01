#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据增强效果对比实验
同时运行增强版和非增强版模型，对比性能差异
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

# 导入原有模块
if __name__ == '__main__':
    from job_offers_classifier.job_offers_classfier_old import (
        ChineseTransformerJobOffersClassifier,
        get_recommended_chinese_models
    )
    from job_offers_classifier.job_offers_utils_old import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    # 导入数据增强模块
    from chinese_job_data_augmentation import EnhancedJobDataProcessor


class AugmentationComparisonExperiment:
    """数据增强效果对比实验类"""
    
    def __init__(self, csv_path: str, max_samples: int = 8000):
        self.csv_path = csv_path
        self.max_samples = max_samples
        
        # 实验配置
        self.test_model = 'hfl/chinese-roberta-wwm-ext'  # 使用最佳模型进行对比
        self.training_config = {
            'max_epochs': 5,
            'patience': 3,
            'max_seq_length': 256,
            'batch_size': 16,
            'learning_rate': 2e-5
        }
        
        print("🔬 数据增强效果对比实验初始化")
        print(f"   测试模型: {self.test_model}")
        print(f"   最大样本数: {self.max_samples}")

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

    def load_original_data(self):
        """加载原始数据（不增强）"""
        print(f"\n📊 加载原始数据（不增强）...")
        
        # 加载CSV
        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding='gbk')
        
        print(f"   原始数据: {len(df)} 行")
        
        # 基础文本组合（简单版本）
        def combine_features(row):
            parts = []
            for col in ['岗位', '岗位描述', '岗位职能']:
                if col in row and pd.notna(row[col]):
                    content = str(row[col])
                    # 处理岗位职能列表格式
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
        
        # 智能采样 - 优先保留多样本类别
        if self.max_samples and len(df) > self.max_samples:
            print(f"   需要从 {len(df)} 行中采样 {self.max_samples} 行")
            
            # 分析类别分布
            class_counts = df['isco_code'].value_counts()
            
            # 优先保留有足够样本的类别
            sufficient_classes = class_counts[class_counts >= 2].index.tolist()
            insufficient_classes = class_counts[class_counts == 1].index.tolist()
            
            print(f"   足够样本类别: {len(sufficient_classes)}")
            print(f"   单样本类别: {len(insufficient_classes)}")
            
            # 分别采样
            sampled_data = []
            
            # 1. 保留所有单样本类别（如果空间允许）
            single_sample_data = df[df['isco_code'].isin(insufficient_classes)]
            remaining_budget = self.max_samples - len(single_sample_data)
            
            if remaining_budget > 0:
                sampled_data.append(single_sample_data)
                
                # 2. 从多样本类别中按比例采样
                multi_sample_data = df[df['isco_code'].isin(sufficient_classes)]
                
                if len(multi_sample_data) > remaining_budget:
                    # 分层采样保持类别平衡
                    try:
                        multi_sample_data = multi_sample_data.groupby('isco_code', group_keys=False).apply(
                            lambda x: x.sample(min(len(x), max(2, remaining_budget // len(sufficient_classes))), 
                                             random_state=42)
                        ).reset_index(drop=True)
                        
                        # 如果还是超出预算，随机采样
                        if len(multi_sample_data) > remaining_budget:
                            multi_sample_data = multi_sample_data.sample(n=remaining_budget, random_state=42)
                            
                    except ValueError:
                        # 如果分层采样失败，直接随机采样
                        multi_sample_data = multi_sample_data.sample(n=remaining_budget, random_state=42)
                
                sampled_data.append(multi_sample_data)
            else:
                # 预算不足，只能从单样本类别中随机选择
                sampled_data.append(single_sample_data.sample(n=self.max_samples, random_state=42))
            
            df = pd.concat(sampled_data, ignore_index=True)
            print(f"   智能采样后样本: {len(df)} 行")
        
        texts = df['combined_text'].tolist()
        labels = df['isco_code'].tolist()
        
        # 最终验证 - 移除仍然单样本的类别（如果必要）
        final_class_counts = pd.Series(labels).value_counts()
        single_classes = final_class_counts[final_class_counts == 1].index.tolist()
        
        if len(single_classes) > len(final_class_counts) * 0.3:  # 如果单样本类别过多（超过30%）
            print(f"   ⚠️ 单样本类别过多({len(single_classes)})，进行最终过滤")
            # 保留多样本类别
            multi_classes = final_class_counts[final_class_counts > 1].index.tolist()
            filtered_data = [(text, label) for text, label in zip(texts, labels) if label in multi_classes]
            
            if len(filtered_data) > 100:  # 确保有足够数据
                texts, labels = zip(*filtered_data)
                texts, labels = list(texts), list(labels)
                print(f"   过滤后样本: {len(texts)} 行")
        
        # 数据统计
        original_stats = {
            'total_samples': len(texts),
            'unique_labels': len(set(labels)),
            'avg_text_length': np.mean([len(text) for text in texts]),
            'label_distribution': pd.Series(labels).value_counts().to_dict()
        }
        
        print(f"✅ 原始数据加载完成:")
        print(f"   样本数: {len(texts)}")
        print(f"   类别数: {len(set(labels))}")
        print(f"   平均文本长度: {original_stats['avg_text_length']:.1f}")
        
        # 最终检查
        final_single_classes = [label for label, count in pd.Series(labels).value_counts().items() if count == 1]
        if final_single_classes:
            print(f"   ⚠️ 仍有 {len(final_single_classes)} 个单样本类别")
        
        return texts, labels, original_stats

    def load_augmented_data(self):
        """加载增强数据"""
        print(f"\n🎯 加载增强数据...")
        
        # 创建数据处理器
        processor = EnhancedJobDataProcessor()
        
        # 处理数据（启用增强和平衡）
        texts, labels, processing_stats = processor.process_csv_data(
            csv_path=self.csv_path,
            enable_augmentation=True,
            balance_data=True,
            target_samples_per_class=6  # 适中的增强量
        )
        
        # 限制样本数（如果增强后超出限制）
        if self.max_samples and len(texts) > self.max_samples * 2:  # 给增强版更多空间
            print(f"   需要从 {len(texts)} 行中采样 {self.max_samples * 2} 行")
            
            # 使用智能采样而不是简单的分层采样
            from collections import Counter
            label_counts = Counter(labels)
            
            # 分析类别分布
            single_classes = [label for label, count in label_counts.items() if count == 1]
            multi_classes = [label for label, count in label_counts.items() if count > 1]
            
            if len(single_classes) > 0:
                print(f"   检测到 {len(single_classes)} 个单样本类别，使用智能采样")
                
                # 保留所有单样本类别
                single_indices = [i for i, label in enumerate(labels) if label in single_classes]
                
                # 从多样本类别中采样
                multi_indices = [i for i, label in enumerate(labels) if label in multi_classes]
                remaining_budget = self.max_samples * 2 - len(single_indices)
                
                if remaining_budget > 0 and len(multi_indices) > remaining_budget:
                    # 分层采样多样本类别
                    multi_texts_labels = [(texts[i], labels[i]) for i in multi_indices]
                    multi_texts_only = [texts[i] for i in multi_indices]
                    multi_labels_only = [labels[i] for i in multi_indices]
                    
                    try:
                        sampled_multi_texts, _, sampled_multi_labels, _ = train_test_split(
                            multi_texts_only, multi_labels_only,
                            train_size=remaining_budget,
                            stratify=multi_labels_only,
                            random_state=42
                        )
                        
                        # 组合结果
                        final_texts = [texts[i] for i in single_indices] + sampled_multi_texts
                        final_labels = [labels[i] for i in single_indices] + sampled_multi_labels
                        
                    except ValueError:
                        # 如果分层采样失败，使用随机采样
                        print("   分层采样失败，使用随机采样")
                        selected_multi_indices = np.random.choice(multi_indices, 
                                                                 size=min(remaining_budget, len(multi_indices)), 
                                                                 replace=False)
                        
                        final_texts = [texts[i] for i in single_indices] + [texts[i] for i in selected_multi_indices]
                        final_labels = [labels[i] for i in single_indices] + [labels[i] for i in selected_multi_indices]
                
                else:
                    # 预算足够或没有多样本类别
                    final_texts = texts
                    final_labels = labels
                
                texts = final_texts
                labels = final_labels
                
            else:
                # 没有单样本类别，正常分层采样
                try:
                    texts, _, labels, _ = train_test_split(
                        texts, labels, 
                        train_size=self.max_samples * 2,
                        stratify=labels,
                        random_state=42
                    )
                except ValueError:
                    # 分层采样失败，随机采样
                    print("   分层采样失败，使用随机采样")
                    indices = np.random.choice(len(texts), size=self.max_samples * 2, replace=False)
                    texts = [texts[i] for i in indices]
                    labels = [labels[i] for i in indices]
            
            print(f"   智能采样后样本: {len(texts)}")
        
        augmented_stats = processing_stats['final_stats']
        
        print(f"✅ 增强数据加载完成:")
        print(f"   原始样本: {processing_stats['original_stats']['total_samples']}")
        print(f"   增强后样本: {len(texts)}")
        print(f"   类别数: {augmented_stats['unique_labels']}")
        print(f"   平均文本长度: {augmented_stats['avg_text_length']:.1f}")
        
        # 最终检查单样本类别
        final_class_counts = Counter(labels)
        final_single_classes = [label for label, count in final_class_counts.items() if count == 1]
        if final_single_classes:
            print(f"   ⚠️ 仍有 {len(final_single_classes)} 个单样本类别")
        
        return texts, labels, augmented_stats

    def train_single_experiment(self, texts, labels, experiment_name, results_dir):
        """训练单个实验"""
        print(f"\n🤖 开始训练: {experiment_name}")
        
        # 智能数据划分 - 处理单样本类别
        print("   📊 分析数据分布...")
        from collections import Counter
        label_counts = Counter(labels)
        
        # 统计单样本类别
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        print(f"   单样本类别: {len(single_sample_classes)}")
        print(f"   多样本类别: {len(multi_sample_classes)}")
        
        if len(single_sample_classes) > 0:
            print(f"   ⚠️ 检测到 {len(single_sample_classes)} 个单样本类别，使用智能划分策略")
            
            # 分别处理单样本和多样本类别
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
                    # 对多样本类别进行分层划分
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_texts)), 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=multi_labels
                    )
                    
                    # 转换回原始索引
                    train_indices.extend([multi_indices[i] for i in multi_train_idx])
                    test_indices.extend([multi_indices[i] for i in multi_test_idx])
            
            # 构建最终的训练测试集
            train_texts = [texts[i] for i in train_indices]
            train_labels = [labels[i] for i in train_indices]
            test_texts = [texts[i] for i in test_indices]
            test_labels = [labels[i] for i in test_indices]
            
        else:
            # 正常分层划分
            train_texts, test_texts, train_labels, test_labels = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
        
        print(f"   训练样本: {len(train_texts)}")
        print(f"   测试样本: {len(test_texts)}")
        
        # 验证数据划分结果
        if len(test_texts) == 0:
            print("   ⚠️ 测试集为空，调整划分策略")
            # 如果测试集为空，强制从训练集中分出一部分
            if len(train_texts) >= 10:
                split_point = max(1, len(train_texts) // 5)  # 取20%作为测试集
                test_texts = train_texts[-split_point:]
                test_labels = train_labels[-split_point:]
                train_texts = train_texts[:-split_point]
                train_labels = train_labels[:-split_point]
                print(f"   调整后 - 训练样本: {len(train_texts)}, 测试样本: {len(test_texts)}")
            else:
                print("   ❌ 数据量太少，无法进行有效训练")
                return {
                    'experiment_name': experiment_name,
                    'model_name': self.test_model,
                    'accuracy': 0.0,
                    'top_3_accuracy': 0.0,
                    'top_5_accuracy': 0.0,
                    'training_time_minutes': 0.0,
                    'status': 'failed',
                    'error': 'Insufficient data for training'
                }
        
        # 创建层次结构
        hierarchy = self.create_isco_hierarchy_from_codes(set(labels))
        
        # 创建模型目录
        model_dir = results_dir / f"model_{experiment_name.lower().replace(' ', '_')}"
        
        start_time = time.time()
        
        try:
            # 创建分类器
            classifier = ChineseTransformerJobOffersClassifier(
                model_dir=str(model_dir),
                hierarchy=hierarchy,
                transformer_model=self.test_model,
                learning_rate=self.training_config['learning_rate'],
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
            val_size = min(300, len(test_texts) // 3)
            val_texts = test_texts[:val_size]
            val_labels = test_labels[:val_size]
            final_test_texts = test_texts[val_size:]
            final_test_labels = test_labels[val_size:]
            
            print(f"   验证样本: {val_size}")
            print(f"   最终测试样本: {len(final_test_texts)}")
            
            # 训练
            print(f"   🎯 开始训练...")
            classifier.fit(train_labels, train_texts, y_val=val_labels, X_val=val_texts)
            
            # 预测
            print(f"   🔮 预测中...")
            predictions_df = classifier.predict(final_test_texts, format='dataframe', top_k=5)
            
            # 计算指标
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
            
            training_time = time.time() - start_time
            
            # 详细分析
            detailed_analysis = self.analyze_predictions(y_true, y_pred, predictions_df)
            
            result = {
                'experiment_name': experiment_name,
                'model_name': self.test_model,
                'train_samples': len(train_texts),
                'test_samples': len(final_test_texts),
                'accuracy': accuracy,
                'top_3_accuracy': top_3_acc,
                'top_5_accuracy': top_5_acc,
                'training_time_minutes': training_time / 60,
                'status': 'success',
                'detailed_analysis': detailed_analysis
            }
            
            print(f"✅ {experiment_name} 训练完成!")
            print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Top-3准确率: {top_3_acc:.4f} ({top_3_acc*100:.2f}%)")
            print(f"   Top-5准确率: {top_5_acc:.4f} ({top_5_acc*100:.2f}%)")
            print(f"   训练时间: {training_time/60:.1f} 分钟")
            
            # 保存详细结果
            detailed_results = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'confidence': predictions_df['prob_1'].tolist(),
                'correct': [t == p for t, p in zip(y_true, y_pred)]
            })
            detailed_results.to_csv(
                results_dir / f"{experiment_name.lower().replace(' ', '_')}_detailed_results.csv", 
                index=False, encoding='utf-8'
            )
            
            return result
            
        except Exception as e:
            print(f"❌ {experiment_name} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'experiment_name': experiment_name,
                'model_name': self.test_model,
                'accuracy': 0.0,
                'top_3_accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'training_time_minutes': 0.0,
                'status': 'failed',
                'error': str(e)
            }

    def analyze_predictions(self, y_true, y_pred, predictions_df):
        """分析预测结果"""
        from collections import defaultdict
        
        # 按类别统计准确率
        class_accuracy = defaultdict(list)
        for true_label, pred_label in zip(y_true, y_pred):
            class_accuracy[true_label].append(true_label == pred_label)
        
        class_acc_summary = {}
        for class_label, correct_list in class_accuracy.items():
            class_acc_summary[class_label] = {
                'accuracy': np.mean(correct_list),
                'samples': len(correct_list)
            }
        
        # 置信度分析
        confidences = predictions_df['prob_1'].tolist()
        correct_mask = [t == p for t, p in zip(y_true, y_pred)]
        
        correct_confidences = [conf for conf, correct in zip(confidences, correct_mask) if correct]
        wrong_confidences = [conf for conf, correct in zip(confidences, correct_mask) if not correct]
        
        return {
            'class_accuracy': class_acc_summary,
            'avg_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0,
            'avg_confidence_wrong': np.mean(wrong_confidences) if wrong_confidences else 0,
            'confidence_stats': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
        }

    def run_comparison(self):
        """运行对比实验"""
        print("🔬 开始数据增强效果对比实验")
        print("=" * 60)
        
        # 创建结果目录
        results_dir = Path("augmentation_comparison_results")
        results_dir.mkdir(exist_ok=True)
        
        experiment_start_time = time.time()
        results = []
        data_stats = {}
        
        # 实验1: 原始数据（不增强）
        print(f"\n{'='*60}")
        print(f"🔵 实验1: 原始数据训练")
        print(f"{'='*60}")
        
        original_texts, original_labels, original_stats = self.load_original_data()
        data_stats['original'] = original_stats
        
        original_result = self.train_single_experiment(
            original_texts, original_labels, "Original Data", results_dir
        )
        results.append(original_result)
        
        # 实验2: 增强数据
        print(f"\n{'='*60}")
        print(f"🟢 实验2: 数据增强训练")
        print(f"{'='*60}")
        
        augmented_texts, augmented_labels, augmented_stats = self.load_augmented_data()
        data_stats['augmented'] = augmented_stats
        
        augmented_result = self.train_single_experiment(
            augmented_texts, augmented_labels, "Augmented Data", results_dir
        )
        results.append(augmented_result)
        
        total_time = time.time() - experiment_start_time
        
        # 生成对比报告
        self.generate_comparison_report(results, data_stats, total_time, results_dir)
        
        return results, results_dir

    def generate_comparison_report(self, results, data_stats, total_time, results_dir):
        """生成对比报告"""
        print(f"\n📊 生成对比分析报告...")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "comparison_results.csv", index=False, encoding='utf-8')
        
        # 计算改进情况
        if len(results) >= 2 and results[0]['status'] == 'success' and results[1]['status'] == 'success':
            original_acc = results[0]['accuracy']
            augmented_acc = results[1]['accuracy']
            improvement = augmented_acc - original_acc
            improvement_pct = (improvement / original_acc * 100) if original_acc > 0 else 0
            
            original_top3 = results[0]['top_3_accuracy']
            augmented_top3 = results[1]['top_3_accuracy']
            top3_improvement = augmented_top3 - original_top3
            top3_improvement_pct = (top3_improvement / original_top3 * 100) if original_top3 > 0 else 0
        else:
            improvement = improvement_pct = 0
            top3_improvement = top3_improvement_pct = 0
        
        # 生成详细报告
        report = {
            'experiment_info': {
                'experiment_type': 'Data Augmentation Effect Comparison',
                'timestamp': datetime.now().isoformat(),
                'total_time_hours': total_time / 3600,
                'model_used': self.test_model,
                'max_samples': self.max_samples
            },
            'data_statistics': data_stats,
            'results': results,
            'comparison_analysis': {
                'accuracy_improvement': improvement,
                'accuracy_improvement_percentage': improvement_pct,
                'top3_improvement': top3_improvement,
                'top3_improvement_percentage': top3_improvement_pct,
                'training_time_difference': results[1]['training_time_minutes'] - results[0]['training_time_minutes'] if len(results) >= 2 else 0
            }
        }
        
        # 保存报告
        with open(results_dir / "comparison_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 打印对比结果
        print(f"\n🎉 数据增强效果对比完成!")
        print(f"📈 实验总结:")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        print(f"   测试模型: {self.test_model}")
        
        if len(results) >= 2:
            print(f"\n📊 详细对比结果:")
            print(f"┌─────────────────┬─────────────┬─────────────┬─────────────┐")
            print(f"│ 实验类型        │ 准确率      │ Top-3准确率 │ 训练时间    │")
            print(f"├─────────────────┼─────────────┼─────────────┼─────────────┤")
            
            for result in results:
                if result['status'] == 'success':
                    name = result['experiment_name'][:15]
                    acc = f"{result['accuracy']:.4f}"
                    top3 = f"{result['top_3_accuracy']:.4f}"
                    time_str = f"{result['training_time_minutes']:.1f}分"
                    print(f"│ {name:<15} │ {acc:<11} │ {top3:<11} │ {time_str:<11} │")
            
            print(f"└─────────────────┴─────────────┴─────────────┴─────────────┘")
            
            if results[0]['status'] == 'success' and results[1]['status'] == 'success':
                print(f"\n🎯 增强效果分析:")
                print(f"   准确率提升: {improvement:+.4f} ({improvement_pct:+.2f}%)")
                print(f"   Top-3准确率提升: {top3_improvement:+.4f} ({top3_improvement_pct:+.2f}%)")
                print(f"   训练时间增加: {report['comparison_analysis']['training_time_difference']:+.1f} 分钟")
                
                if improvement > 0:
                    print(f"   🎉 数据增强效果显著！")
                elif improvement > -0.01:
                    print(f"   ⚖️ 数据增强效果中性")
                else:
                    print(f"   ⚠️ 数据增强可能需要调优")
                
                # 数据量对比
                print(f"\n📊 数据量对比:")
                original_samples = data_stats['original']['total_samples']
                augmented_samples = data_stats['augmented']['total_samples']
                increase_ratio = augmented_samples / original_samples
                print(f"   原始数据: {original_samples:,} 样本")
                print(f"   增强数据: {augmented_samples:,} 样本")
                print(f"   增长倍数: {increase_ratio:.1f}x")
                
        print(f"\n📁 详细结果保存在: {results_dir}")
        
        # 生成可视化建议
        print(f"\n💡 后续分析建议:")
        print(f"   1. 查看 comparison_results.csv 了解详细指标")
        print(f"   2. 查看 *_detailed_results.csv 分析具体预测")
        print(f"   3. 如果效果不佳，可以调整增强策略参数")
        print(f"   4. 考虑在更大数据集上验证效果")


def main():
    """主函数"""
    print("🔬 数据增强效果对比实验")
    print("比较原始数据 vs 增强数据的模型性能")
    print("=" * 60)
    
    # 配置实验
    csv_path = "newjob1_sortall.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        print("请确保数据文件在当前目录下")
        return
    
    print("选择实验规模:")
    print("1. 快速对比 (8K样本限制)")
    print("2. 标准对比 (15K样本限制)")
    print("3. 完整对比 (无样本限制)")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == "1":
        max_samples = 8000
    elif choice == "2":
        max_samples = 15000
    elif choice == "3":
        max_samples = None
    else:
        print("无效选择，使用快速对比")
        max_samples = 8000
    
    print(f"\n🎯 实验配置:")
    print(f"   数据文件: {csv_path}")
    print(f"   样本限制: {max_samples if max_samples else '无限制'}")
    print(f"   测试模型: hfl/chinese-roberta-wwm-ext")
    print(f"   对比内容: 原始数据 vs 数据增强")
    
    confirm = input(f"\n确认开始对比实验? (y/N): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return
    
    try:
        # 创建实验对象
        experiment = AugmentationComparisonExperiment(csv_path, max_samples)
        
        # 运行对比实验
        results, results_dir = experiment.run_comparison()
        
        print(f"\n🎯 对比实验完成! 查看详细结果: {results_dir}")
        
    except Exception as e:
        print(f"❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()