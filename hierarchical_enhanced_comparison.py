#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版层次化损失函数对比实验
集成中文预处理和智能数据划分
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
import torch
from collections import Counter

warnings.filterwarnings('ignore')

# 导入原有模块
if __name__ == '__main__':
    # 确保可以找到层次化工具
    sys.path.append('.')
    
    # 尝试导入层次化功能
    try:
        from hierarchical_utils import create_hierarchical_components, HierarchicalLoss
        HIERARCHICAL_AVAILABLE = True
        print("✅ 层次化功能可用")
    except ImportError as e:
        HIERARCHICAL_AVAILABLE = False
        print(f"⚠️ 层次化功能不可用: {e}")
    
    from job_offers_classifier.job_offers_classfier_old import (
        ChineseTransformerJobOffersClassifier,
        get_recommended_chinese_models
    )
    from job_offers_classifier.job_offers_utils_old import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    
    from chinese_job_data_augmentation import EnhancedJobDataProcessor


class HierarchicalEnhancedClassifier(ChineseTransformerJobOffersClassifier):
    """
    在现有分类器基础上添加层次化功能的包装类
    """
    
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 ckpt_path=None,
                 transformer_model="hfl/chinese-roberta-wwm-ext",
                 transformer_ckpt_path="",
                 modeling_mode='bottom-up',
                 adam_epsilon=1e-8,
                 learning_rate=2e-5,
                 weight_decay=0.01,
                 max_epochs=20,
                 batch_size=64,
                 max_sequence_length=256,
                 early_stopping=True,
                 early_stopping_delta=0.001,
                 early_stopping_patience=3,
                 devices=1,
                 accelerator="auto",
                 num_nodes=1,
                 threads=-1,
                 precision=16,
                 verbose=True,
                 # 新增层次化参数
                 use_hierarchical_loss=True,
                 use_multitask_learning=False,
                 hierarchical_loss_weights=None,
                 task_weights=None):
        
        # 调用父类初始化
        super().__init__(
            model_dir=model_dir,
            hierarchy=hierarchy,
            ckpt_path=ckpt_path,
            transformer_model=transformer_model,
            transformer_ckpt_path=transformer_ckpt_path,
            modeling_mode=modeling_mode,
            adam_epsilon=adam_epsilon,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            batch_size=batch_size,
            max_sequence_length=max_sequence_length,
            early_stopping=early_stopping,
            early_stopping_delta=early_stopping_delta,
            early_stopping_patience=early_stopping_patience,
            devices=devices,
            accelerator=accelerator,
            num_nodes=num_nodes,
            threads=threads,
            precision=precision,
            verbose=verbose
        )
        
        # 层次化功能配置
        self.hierarchical_available = HIERARCHICAL_AVAILABLE and hierarchy is not None
        self.use_hierarchical_loss = use_hierarchical_loss and self.hierarchical_available
        self.use_multitask_learning = use_multitask_learning and self.hierarchical_available
        
        # 默认权重配置
        self.hierarchical_loss_weights = hierarchical_loss_weights or {1: 1, 2:1.0, 3: 1, 4:1}
        self.task_weights = task_weights or {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        
        if self.verbose and self.hierarchical_available:
            print(f"🔧 层次化增强分类器初始化")
            print(f"   层次化损失: {'✓' if self.use_hierarchical_loss else '✗'}")
            print(f"   多任务学习: {'✓' if self.use_multitask_learning else '✗'}")
            if self.use_hierarchical_loss:
                print(f"   层次权重: {self.hierarchical_loss_weights}")

    def get_hierarchical_performance(self, X, y_true_codes):
        """获取层次化性能评估"""
        if not self.use_hierarchical_loss:
            return None
            
        # 预测
        predictions_df = self.predict(X, format='dataframe', top_k=5)
        y_pred_codes = predictions_df['class_1'].tolist()
        
        # 计算各级别准确率
        level_accuracies = {}
        
        for level in [1, 2, 3, 4]:
            true_level_codes = [code[:level] for code in y_true_codes]
            pred_level_codes = [code[:level] for code in y_pred_codes]
            
            level_acc = accuracy_score(true_level_codes, pred_level_codes)
            level_accuracies[f'level_{level}_accuracy'] = level_acc
        
        return {
            'level_accuracies': level_accuracies,
            'predictions': predictions_df,
            'hierarchical_error_analysis': self._analyze_hierarchical_errors(y_true_codes, y_pred_codes)
        }

    def _analyze_hierarchical_errors(self, y_true_codes, y_pred_codes):
        """分析层次化错误"""
        error_analysis = {
            'same_1_level': 0,  # 1级相同但更细级别错误
            'same_2_level': 0,  # 2级相同但更细级别错误  
            'same_3_level': 0,  # 3级相同但4级错误
            'different_1_level': 0,  # 1级就不同（最严重错误）
            'total_errors': 0
        }
        
        for true_code, pred_code in zip(y_true_codes, y_pred_codes):
            if true_code != pred_code:
                error_analysis['total_errors'] += 1
                
                if true_code[:1] != pred_code[:1]:
                    error_analysis['different_1_level'] += 1
                elif true_code[:2] != pred_code[:2]:
                    error_analysis['same_1_level'] += 1
                elif true_code[:3] != pred_code[:3]:
                    error_analysis['same_2_level'] += 1
                else:
                    error_analysis['same_3_level'] += 1
        
        return error_analysis


class HierarchicalExperimentRunner:
    """层次化实验运行器 - 集成中文预处理"""
    
    def __init__(self, csv_path: str, max_samples: int = 8000):
        self.csv_path = csv_path
        self.max_samples = max_samples
        self._data_cache = {}
        # 测试配置
        # 选项A：快速测试（推荐）
        self.test_models = [
            'hfl/chinese-roberta-wwm-ext'  # 只测试最好的模型
        ]
        
        # 选项B：完整测试
        # self.test_models = [
        #     'bert-base-chinese',
        #     'hfl/chinese-bert-wwm-ext',
        #     'hfl/chinese-roberta-wwm-ext'
        # ]
        
      
        self.training_config = {
            'max_epochs': 3,
            'patience': 2,
            'max_seq_length': 256,
            'batch_size': 64,
            'learning_rate': 2e-5
        }
        
        print("🔬 层次化增强实验初始化")
        print(f"   层次化功能: {'✓' if HIERARCHICAL_AVAILABLE else '✗'}")
        print(f"   中文预处理: ✓")

    def load_enhanced_data(self, enable_augmentation=True, target_samples_per_class=6):
        """加载增强版中文预处理数据 - 带缓存优化"""
        
        # 🚀 检查缓存
        cache_key = f"aug_{enable_augmentation}_samples_{target_samples_per_class}_max_{self.max_samples}"
        if cache_key in self._data_cache:
            print(f"✅ 使用缓存的预处理数据: {cache_key}")
            return self._data_cache[cache_key]
        
        # 原有的数据处理逻辑...
        data_type = "增强预处理" if enable_augmentation else "基础预处理"
        print(f"\n📊 加载{data_type}数据...")
        
        if enable_augmentation:
            processor = EnhancedJobDataProcessor()
            texts, labels, processing_stats = processor.process_csv_data(
                csv_path=self.csv_path,
                enable_augmentation=True,
                balance_data=True,
                target_samples_per_class=target_samples_per_class
            )
        else:
            processor = EnhancedJobDataProcessor()
            texts, labels, processing_stats = processor.process_csv_data(
                csv_path=self.csv_path,
                enable_augmentation=False,
                balance_data=False,
                target_samples_per_class=1
            )
        
        # 限制样本数（如果需要）
        if self.max_samples and len(texts) > self.max_samples:
            print(f"   采样 {self.max_samples} 行从 {len(texts)} 行")
            indices = np.random.choice(len(texts), size=self.max_samples, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
        
        stats = processing_stats['final_stats']
        result = (texts, labels, stats)
        
        # 🚀 缓存结果
        self._data_cache[cache_key] = result
        print(f"✅ 数据已缓存: {cache_key}")
        
        return result

    def safe_train_test_split_with_hierarchy(self, texts, labels):
        """考虑ISCO层次结构的安全数据划分"""
        print("   📊 智能数据划分中...")
        
        # 分析类别分布
        label_counts = Counter(labels)
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        print(f"   4级编码 - 单样本类别: {len(single_sample_classes)}, 多样本类别: {len(multi_sample_classes)}")
        
        # 分析各级别分布
        for level in [1, 2, 3]:
            level_codes = [label[:level] for label in labels]
            level_counts = Counter(level_codes)
            level_single = sum(1 for count in level_counts.values() if count == 1)
            print(f"   {level}级编码 - 单样本类别: {level_single}/{len(level_counts)}")
        
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
                    # 尝试分层划分
                    multi_train_idx, multi_test_idx = train_test_split(
                        range(len(multi_texts)), 
                        test_size=0.2, 
                        random_state=42, 
                        stratify=multi_labels
                    )
                    print(f"   ✓ 成功进行分层划分")
                except ValueError as e:
                    print(f"   ⚠️ 分层划分失败，使用随机划分: {e}")
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
            print(f"   ⚠️ 测试集为空，从训练集中分出一部分")
            split_point = max(1, len(train_texts) // 5)
            test_texts = train_texts[-split_point:]
            test_labels = train_labels[-split_point:]
            train_texts = train_texts[:-split_point]
            train_labels = train_labels[:-split_point]
        
        print(f"   ✓ 最终划分 - 训练: {len(train_texts)}, 测试: {len(test_texts)}")
        
        # 最终验证：确保测试集中没有单样本类别
        if len(test_texts) > 0:
            test_label_counts = Counter(test_labels)
            test_single_classes = [label for label, count in test_label_counts.items() if count == 1]
            if test_single_classes:
                print(f"   ⚠️ 测试集中仍有 {len(test_single_classes)} 个单样本类别")
        
        return train_texts, test_texts, train_labels, test_labels

    def run_hierarchical_comparison(self):
        """运行层次化功能对比实验 - 优化版"""
        print("🔬 开始优化版层次化功能对比实验")
        print("🚀 优化项：增大batch_size, 减少epoch, 激进早停, 数据缓存")
        print("=" * 80)
        
        results_dir = Path("hierarchical_comparison_results_optimized")
        results_dir.mkdir(exist_ok=True)
        
        all_results = []
        
        # 🚀 先只测试增强预处理（通常效果更好）
        data_configs = [
            {"enable_augmentation": True, "name": "增强预处理", "target_samples": 6}
            # 如果增强预处理效果好，再测试基础预处理
        ]
        
        for data_config in data_configs:
            print(f"\n{'='*80}")
            print(f"📊 数据处理方式: {data_config['name']}")
            print(f"{'='*80}")
            
            # 🚀 预加载并缓存数据
            texts, labels, stats = self.load_enhanced_data(
                enable_augmentation=data_config["enable_augmentation"],
                target_samples_per_class=data_config["target_samples"]
            )
            
            # 对每个模型运行对比实验
            for model_name in self.test_models:
                print(f"\n{'='*60}")
                print(f"🤖 测试模型: {model_name}")
                print(f"{'='*60}")
                
                # 实验1: 标准损失函数
                standard_result = self._run_single_experiment(
                    model_name, texts, labels, 
                    f"Standard Loss ({data_config['name']})", 
                    results_dir,
                    use_hierarchical_loss=False
                )
                all_results.append(standard_result)
                
                # 实验2: 层次化损失函数
                if HIERARCHICAL_AVAILABLE:
                    hierarchical_result = self._run_single_experiment(
                        model_name, texts, labels, 
                        f"Hierarchical Loss ({data_config['name']})", 
                        results_dir,
                        use_hierarchical_loss=True
                    )
                    all_results.append(hierarchical_result)
                    
                    # 打印对比结果
                    if standard_result['status'] == 'success' and hierarchical_result['status'] == 'success':
                        self._print_comparison(standard_result, hierarchical_result, model_name, data_config['name'])
                else:
                    print("⚠️ 层次化功能不可用,跳过层次化损失实验")
        
        # 生成报告
        self._generate_hierarchical_report(all_results, results_dir)
        
        return all_results, results_dir


    def _run_single_experiment(self, model_name, texts, labels, experiment_name, results_dir, 
                            use_hierarchical_loss=False):
        """运行单个实验"""
        print(f"\n🎯 {experiment_name} - {model_name}")
        
        try:
            # 使用智能数据分割
            train_texts, test_texts, train_labels, test_labels = self.safe_train_test_split_with_hierarchy(texts, labels)
            
            if len(test_texts) < 5:
                print(f"   ❌ 测试集样本不足: {len(test_texts)}")
                return {
                    'model_name': model_name,
                    'experiment_name': experiment_name,
                    'use_hierarchical_loss': use_hierarchical_loss,
                    'status': 'failed',
                    'error': f'Insufficient test data: {len(test_texts)} samples'
                }
            
            # ⭐ 添加验证集划分
            val_size = min(200, len(test_texts) // 2)  # 从测试集中分出一部分作为验证集
            if val_size > 0 and len(test_texts) > val_size:
                val_texts = test_texts[:val_size]
                val_labels = test_labels[:val_size]
                final_test_texts = test_texts[val_size:]
                final_test_labels = test_labels[val_size:]
            else:
                val_texts, val_labels = None, None
                final_test_texts = test_texts
                final_test_labels = test_labels
            
            # 创建层次结构
            hierarchy = self.create_isco_hierarchy_from_codes(set(labels))
            
            # 模型目录
            safe_model_name = model_name.replace('/', '_').replace('-', '_')
            safe_exp_name = experiment_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            model_dir = results_dir / f"model_{safe_model_name}_{safe_exp_name}"
            
            start_time = time.time()
            
            # 创建分类器
            if use_hierarchical_loss and HIERARCHICAL_AVAILABLE:
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
                    precision="16-mixed" if torch.cuda.is_available() else 32,
                    threads=0,
                    verbose=True,
                    # ⭐ 显式传递层次化参数
                    use_hierarchical_loss=True,
                    hierarchical_loss_weights={1: 6, 2: 0.01, 3: 0.01, 4: 0.01}
                )
                print(f"   ✓ 创建层次化增强分类器")
            else:
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
                    precision="16-mixed" if torch.cuda.is_available() else 32,
                    threads=0,
                    verbose=True,
                    # ⭐ 显式传递层次化参数（关闭）
                    use_hierarchical_loss=False,
                    use_multitask_learning=False
                )
                print(f"   ✓ 创建标准分类器")
            
            print(f"   训练样本: {len(train_texts)}")
            print(f"   验证样本: {len(val_texts) if val_texts else 0}")
            print(f"   测试样本: {len(final_test_texts)}")
            
            # 训练（包含验证集）
            print(f"   🎯 开始训练...")
            if val_texts and val_labels:
                classifier.fit(train_labels, train_texts, y_val=val_labels, X_val=val_texts)
            else:
                classifier.fit(train_labels, train_texts)
            
            # 预测和评估
            print(f"   🔮 预测评估中...")
            predictions_df = classifier.predict(final_test_texts, format='dataframe', top_k=5)
            
            # 其余代码保持不变...
                
           # ⭐ 修复：确保预测结果和标签数量一致
            if len(predictions_df) != len(final_test_labels):
                print(f"   ⚠️ 预测结果数量不匹配: 预测{len(predictions_df)}, 标签{len(final_test_labels)}")
                min_len = min(len(predictions_df), len(final_test_labels))
                predictions_df = predictions_df.iloc[:min_len]
                final_test_labels = final_test_labels[:min_len]
                print(f"   📏 调整后数量: {min_len}")

            # 基础指标
            y_true = final_test_labels
            y_pred = predictions_df['class_1'].tolist()

            # 🌟 新增：计算层次化准确率（1-4级）
            hierarchical_accuracies = {}
            for level in [1, 2, 3, 4]:
                # 截取到对应级别
                y_true_level = [str(code)[:level] for code in y_true]
                y_pred_level = [str(code)[:level] for code in y_pred]
                
                level_accuracy = accuracy_score(y_true_level, y_pred_level)
                hierarchical_accuracies[f'level_{level}_accuracy'] = level_accuracy
            
            # 原有的4级准确率
            accuracy_4_level = hierarchical_accuracies['level_4_accuracy']
            
            # 🌟 新增：计算层次化Top-k准确率
            hierarchical_top_k = {}
            for k in [3, 5]:
                level_top_k = {}
                for level in [1, 2, 3, 4]:
                    level_correct = 0
                    for i, true_label in enumerate(y_true):
                        true_level_code = str(true_label)[:level]
                        # 检查top-k预测中是否有匹配的级别
                        for j in range(1, k+1):
                            if f'class_{j}' in predictions_df.columns:
                                pred_code = str(predictions_df.iloc[i][f'class_{j}'])[:level]
                                if true_level_code == pred_code:
                                    level_correct += 1
                                    break
                    level_top_k[f'level_{level}'] = level_correct / len(y_true)
                hierarchical_top_k[f'top_{k}'] = level_top_k
            
            # 🌟 新增：层次化错误分析（带异常处理）
            try:
                hierarchical_error_analysis = self._analyze_hierarchical_errors_detailed(y_true, y_pred)
            except Exception as e:
                print(f"   ⚠️ 层次化错误分析失败: {e}")
                hierarchical_error_analysis = {
                    'total_samples': len(y_true),
                    'error': str(e)
                }
            
            # 计算传统Top-k准确率（用于对比）
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
                'experiment_name': experiment_name,
                'use_hierarchical_loss': use_hierarchical_loss,
                'train_samples': len(train_texts),
                'test_samples': len(final_test_texts),
                
                # 🌟 层次化准确率（核心指标）
                'level_1_accuracy': hierarchical_accuracies['level_1_accuracy'],
                'level_2_accuracy': hierarchical_accuracies['level_2_accuracy'], 
                'level_3_accuracy': hierarchical_accuracies['level_3_accuracy'],
                'level_4_accuracy': hierarchical_accuracies['level_4_accuracy'],
                
                # 传统指标（保持兼容性）
                'accuracy': accuracy_4_level,  # 等同于level_4_accuracy
                'top_3_accuracy': top_3_acc,
                'top_5_accuracy': top_5_acc,
                
                # 🌟 层次化Top-k准确率
                'hierarchical_top_3': hierarchical_top_k['top_3'],
                'hierarchical_top_5': hierarchical_top_k['top_5'],
                
                # 🌟 层次化错误分析
                'hierarchical_error_analysis': hierarchical_error_analysis,
                
                'training_time_minutes': training_time / 60,
                'status': 'success'
            }
            
            print(f"✅ {experiment_name} 完成!")
            print(f"\n🎯 层次化准确率评估:")
            print(f"   1级准确率: {hierarchical_accuracies['level_1_accuracy']:.4f} ({hierarchical_accuracies['level_1_accuracy']*100:.2f}%)")
            print(f"   2级准确率: {hierarchical_accuracies['level_2_accuracy']:.4f} ({hierarchical_accuracies['level_2_accuracy']*100:.2f}%)")
            print(f"   3级准确率: {hierarchical_accuracies['level_3_accuracy']:.4f} ({hierarchical_accuracies['level_3_accuracy']*100:.2f}%)")
            print(f"   4级准确率: {hierarchical_accuracies['level_4_accuracy']:.4f} ({hierarchical_accuracies['level_4_accuracy']*100:.2f}%)")
            
            print(f"\n📊 传统Top-k准确率:")
            print(f"   Top-3准确率: {top_3_acc:.4f} ({top_3_acc*100:.2f}%)")
            print(f"   Top-5准确率: {top_5_acc:.4f} ({top_5_acc*100:.2f}%)")
            
            print(f"\n🌟 层次化Top-3准确率:")
            for level in [1, 2, 3, 4]:
                acc = hierarchical_top_k['top_3'][f'level_{level}']
                print(f"   {level}级Top-3: {acc:.4f} ({acc*100:.2f}%)")
            
            print(f"\n⏱️ 训练时间: {training_time/60:.1f} 分钟")
            
            # 🌟 显示层次化错误分析（带异常处理）
            if 'error' not in hierarchical_error_analysis:
                self._print_hierarchical_error_analysis(hierarchical_error_analysis)
            
            return result
            
        except Exception as e:
            print(f"❌ {experiment_name} 失败: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'model_name': model_name,
                'experiment_name': experiment_name,
                'use_hierarchical_loss': use_hierarchical_loss,
                'status': 'failed',
                'error': str(e)
            }
    def _analyze_hierarchical_errors_detailed(self, y_true_codes, y_pred_codes):
        """详细的层次化错误分析 - 修复版"""
        error_analysis = {
            'total_samples': len(y_true_codes),
            'correct_samples': 0,
            'error_distribution': {
                'level_1_errors': 0,  # 1级就错误（最严重）
                'level_2_errors': 0,  # 1级对，2级错误
                'level_3_errors': 0,  # 1-2级对，3级错误
                'level_4_errors': 0,  # 1-3级对，4级错误
            },
            'level_agreements': {
                'level_1_agreement': 0,  # 1级相同的样本数
                'level_2_agreement': 0,  # 1-2级相同的样本数
                'level_3_agreement': 0,  # 1-3级相同的样本数
                'level_4_agreement': 0,  # 完全相同的样本数
            }
        }
        
        for true_code, pred_code in zip(y_true_codes, y_pred_codes):
            true_str = str(true_code)
            pred_str = str(pred_code)
            
            # 检查各级别的一致性
            level_matches = []
            for level in [1, 2, 3, 4]:
                true_level = true_str[:level]
                pred_level = pred_str[:level]
                level_matches.append(true_level == pred_level)
            
            # 统计各级别同意度
            if level_matches[0]:  # 1级相同
                error_analysis['level_agreements']['level_1_agreement'] += 1
            if all(level_matches[:2]):  # 1-2级相同
                error_analysis['level_agreements']['level_2_agreement'] += 1
            if all(level_matches[:3]):  # 1-3级相同
                error_analysis['level_agreements']['level_3_agreement'] += 1
            if all(level_matches):  # 完全相同
                error_analysis['level_agreements']['level_4_agreement'] += 1
                error_analysis['correct_samples'] += 1
            else:
                # 分析错误类型
                if not level_matches[0]:
                    error_analysis['error_distribution']['level_1_errors'] += 1
                elif not level_matches[1]:
                    error_analysis['error_distribution']['level_2_errors'] += 1
                elif not level_matches[2]:
                    error_analysis['error_distribution']['level_3_errors'] += 1
                else:
                    error_analysis['error_distribution']['level_4_errors'] += 1
        
        # 🔧 修复：分别计算比例，避免字典迭代时修改
        total = error_analysis['total_samples']
        
        # 先收集需要添加的键值对
        agreement_rates = {}
        for key in list(error_analysis['level_agreements'].keys()):  # 使用list()创建副本
            rate_key = key + '_rate'
            agreement_rates[rate_key] = error_analysis['level_agreements'][key] / total
        
        error_rates = {}
        for key in list(error_analysis['error_distribution'].keys()):  # 使用list()创建副本
            rate_key = key + '_rate'
            error_rates[rate_key] = error_analysis['error_distribution'][key] / total
        
        # 然后一次性添加所有新键
        error_analysis['level_agreements'].update(agreement_rates)
        error_analysis['error_distribution'].update(error_rates)
        
        return error_analysis


    def _print_hierarchical_error_analysis(self, error_analysis):
        """打印层次化错误分析"""
        print(f"\n🔍 层次化错误分析:")
        print(f"   总样本数: {error_analysis['total_samples']}")
        print(f"   完全正确: {error_analysis['correct_samples']} ({error_analysis['correct_samples']/error_analysis['total_samples']*100:.1f}%)")
        
        print(f"\n📊 各级别同意度:")
        agreements = error_analysis['level_agreements']
        for level in [1, 2, 3, 4]:
            count = agreements[f'level_{level}_agreement']
            rate = agreements[f'level_{level}_agreement_rate']
            print(f"   {level}级同意: {count} ({rate*100:.1f}%)")
        
        print(f"\n❌ 错误分布:")
        errors = error_analysis['error_distribution']
        for level in [1, 2, 3, 4]:
            count = errors[f'level_{level}_errors']
            rate = errors[f'level_{level}_errors_rate']
            error_desc = {
                1: "1级就错误(最严重)",
                2: "1级对,2级错",  
                3: "1-2级对,3级错",
                4: "1-3级对,仅4级错"
            }
            print(f"   {error_desc[level]}: {count} ({rate*100:.1f}%)")
    def _print_comparison(self, standard_result, hierarchical_result, model_name, data_config_name):
        """打印层次化对比结果 - 重点关注各级别改进"""
        print(f"\n📊 {model_name} 层次化对比结果 ({data_config_name}):")
        
        # 🌟 层次化准确率对比（核心）
        print(f"\n🎯 各级别准确率对比:")
        for level in [1, 2, 3, 4]:
            std_acc = standard_result[f'level_{level}_accuracy']
            hier_acc = hierarchical_result[f'level_{level}_accuracy']
            improvement = hier_acc - std_acc
            
            print(f"   {level}级: {std_acc:.4f} → {hier_acc:.4f} ({improvement:+.4f}, {improvement*100:+.2f}%)")
        
        # 🌟 层次化价值分析
        print(f"\n💡 层次化价值分析:")
        
        # 计算层次化改进权重分数
        level_weights = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}  # 粗粒度更重要
        weighted_improvement = 0
        for level in [1, 2, 3, 4]:
            std_acc = standard_result[f'level_{level}_accuracy']
            hier_acc = hierarchical_result[f'level_{level}_accuracy']
            improvement = hier_acc - std_acc
            weighted_improvement += improvement * level_weights[level]
        
        print(f"   加权改进分数: {weighted_improvement:+.4f} (权重: 1级=4.0, 2级=3.0, 3级=2.0, 4级=1.0)")
        
        # 🌟 错误严重性分析
        if 'hierarchical_error_analysis' in hierarchical_result:
            std_errors = standard_result.get('hierarchical_error_analysis', {}).get('error_distribution', {})
            hier_errors = hierarchical_result['hierarchical_error_analysis']['error_distribution']
            
            print(f"\n🔍 错误严重性改进:")
            error_types = {
                'level_1_errors': '1级错误(最严重)',
                'level_2_errors': '2级错误', 
                'level_3_errors': '3级错误',
                'level_4_errors': '4级错误(最轻微)'
            }
            
            for error_type, desc in error_types.items():
                std_rate = std_errors.get(f'{error_type}_rate', 0)
                hier_rate = hier_errors.get(f'{error_type}_rate', 0)
                reduction = std_rate - hier_rate
                print(f"   {desc}: {std_rate:.3f} → {hier_rate:.3f} ({reduction:+.3f})")
        
        # 传统指标对比
        std_acc_4 = standard_result['accuracy']
        hier_acc_4 = hierarchical_result['accuracy']
        improvement_4 = hier_acc_4 - std_acc_4
        
        print(f"\n📈 传统指标对比:")
        print(f"   4级准确率: {std_acc_4:.4f} → {hier_acc_4:.4f} ({improvement_4:+.4f}, {improvement_4*100:+.2f}%)")
        print(f"   Top-3: {standard_result['top_3_accuracy']:.4f} → {hierarchical_result['top_3_accuracy']:.4f}")
        print(f"   Top-5: {standard_result['top_5_accuracy']:.4f} → {hierarchical_result['top_5_accuracy']:.4f}")
        
        # 🌟 层次化效果评判
        print(f"\n🎉 层次化效果评判:")
        if weighted_improvement > 0.02:
            print(f"   ✅ 层次化损失显著改善分类质量!")
        elif weighted_improvement > 0.01:
            print(f"   ✓ 层次化损失明显改善分类质量")
        elif weighted_improvement > 0:
            print(f"   ⚖️ 层次化损失轻微改善分类质量")
        else:
            print(f"   ⚠️ 层次化损失需要进一步调优")
        
        # 具体建议
        level_1_improvement = hierarchical_result['level_1_accuracy'] - standard_result['level_1_accuracy']
        level_4_improvement = hierarchical_result['level_4_accuracy'] - standard_result['level_4_accuracy']
        
        if level_1_improvement > 0.01 and level_4_improvement < 0:
            print(f"   💡 成功实现\"牺牲细节换取基础准确性\"的目标")
        elif level_1_improvement > 0 and level_4_improvement >= 0:
            print(f"   🎯 实现了双赢：既提升基础又保持细节")
        elif level_1_improvement < 0:
            print(f"   ⚠️ 建议调整层次权重，加强对粗粒度分类的重视")

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

# 在你的代码中找到 _generate_hierarchical_report 方法，替换其中的一部分

    def _generate_hierarchical_report(self, results, results_dir):
        """生成增强的层次化实验报告"""
        print(f"\n📊 生成层次化实验报告...")
        
        # 保存原始结果
        results_df = pd.DataFrame(results)
        results_df.to_csv(results_dir / "hierarchical_comparison_results.csv", index=False)
        
        # 分析结果
        successful_results = [r for r in results if r['status'] == 'success']
        
        if len(successful_results) == 0:
            print("❌ 没有成功的实验结果")
            return
        
        # 🌟 层次化改进分析
        hierarchical_improvements = []
        
        for model_name in self.test_models:
            model_results = [r for r in successful_results if r['model_name'] == model_name]
            
            for data_type in ['基础预处理', '增强预处理']:
                type_results = [r for r in model_results if data_type in r['experiment_name']]
                
                standard = next((r for r in type_results if not r['use_hierarchical_loss']), None)
                hierarchical = next((r for r in type_results if r['use_hierarchical_loss']), None)
                
                if standard and hierarchical:
                    # 🌟 计算各级别改进
                    level_improvements = {}
                    weighted_improvement = 0
                    level_weights = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0}
                    
                    for level in [1, 2, 3, 4]:
                        std_acc = standard[f'level_{level}_accuracy']
                        hier_acc = hierarchical[f'level_{level}_accuracy']
                        improvement = hier_acc - std_acc
                        level_improvements[f'level_{level}_improvement'] = improvement
                        weighted_improvement += improvement * level_weights[level]
                    
                    hierarchical_improvements.append({
                        'model_name': model_name,
                        'data_type': data_type,
                        'weighted_improvement': weighted_improvement,
                        **level_improvements,
                        'standard_level_1': standard['level_1_accuracy'],
                        'hierarchical_level_1': hierarchical['level_1_accuracy'],
                        'standard_level_4': standard['level_4_accuracy'], 
                        'hierarchical_level_4': hierarchical['level_4_accuracy'],
                    })
        
        # 🌟 生成增强报告
        report = {
            'experiment_info': {
                'experiment_type': 'Enhanced Hierarchical Loss Comparison',
                'timestamp': datetime.now().isoformat(),
                'focus': 'Multi-level accuracy analysis',
                'evaluation_philosophy': 'Prioritize coarse-grained accuracy over fine-grained',
                'hierarchical_available': HIERARCHICAL_AVAILABLE,
                'models_tested': self.test_models,
            },
            'hierarchical_results': hierarchical_improvements,
            'summary': {
                'avg_weighted_improvement': np.mean([imp['weighted_improvement'] for imp in hierarchical_improvements]) if hierarchical_improvements else 0,
                'avg_level_1_improvement': np.mean([imp['level_1_improvement'] for imp in hierarchical_improvements]) if hierarchical_improvements else 0,
                'avg_level_4_improvement': np.mean([imp['level_4_improvement'] for imp in hierarchical_improvements]) if hierarchical_improvements else 0,
                'best_hierarchical_combo': max(hierarchical_improvements, key=lambda x: x['weighted_improvement']) if hierarchical_improvements else None
            },
            'raw_results': results
        }
        
        with open(results_dir / "enhanced_hierarchical_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        # 🌟 打印增强分析
        print(f"\n🎉 增强层次化分析完成!")
        
        if hierarchical_improvements:
            print(f"\n🎯 层次化损失函数效果分析:")
            
            summary = report['summary']
            print(f"   平均加权改进: {summary['avg_weighted_improvement']:+.4f}")
            print(f"   平均1级改进: {summary['avg_level_1_improvement']:+.4f} ({summary['avg_level_1_improvement']*100:+.2f}%)")
            print(f"   平均4级改进: {summary['avg_level_4_improvement']:+.4f} ({summary['avg_level_4_improvement']*100:+.2f}%)")
            
            # 按模型显示
            print(f"\n🔹 各模型表现:")
            for imp in hierarchical_improvements:
                model_short = imp['model_name'].split('/')[-1]
                print(f"   {model_short} ({imp['data_type']}):")
                print(f"     加权改进: {imp['weighted_improvement']:+.4f}")
                print(f"     1级: {imp['level_1_improvement']:+.4f}, 4级: {imp['level_4_improvement']:+.4f}")
            
            # 🌟 效果总结
            avg_weighted = summary['avg_weighted_improvement']
            avg_level_1 = summary['avg_level_1_improvement']
            avg_level_4 = summary['avg_level_4_improvement']
            
            print(f"\n💡 层次化价值总结:")
            if avg_weighted > 0.02:
                print(f"   🎉 层次化损失显著提升整体分类质量!")
            elif avg_weighted > 0.01:
                print(f"   ✅ 层次化损失明显改善分类层次性")
            elif avg_weighted > 0:
                print(f"   ⚖️ 层次化损失有轻微改善")
            else:
                print(f"   ⚠️ 层次化损失需要调优权重配置")
            
            if avg_level_1 > 0.01 and avg_level_4 < 0:
                print(f"   🎯 成功实现'牺牲细节换取基础准确性'策略")
            elif avg_level_1 > 0 and avg_level_4 > 0:
                print(f"   🏆 实现双赢：基础和细节准确性都有提升")
            
            # 最佳组合
            if summary['best_hierarchical_combo']:
                best = summary['best_hierarchical_combo']
                best_model_short = best['model_name'].split('/')[-1]
                print(f"   🏅 最佳组合: {best_model_short} + {best['data_type']}")
                print(f"      加权改进: {best['weighted_improvement']:+.4f}")

        print(f"\n📁 详细结果保存在: {results_dir}")
        print(f"   - enhanced_hierarchical_report.json: 层次化分析报告")
        print(f"   - hierarchical_comparison_results.csv: 原始数据")
def main():
    """主函数 - 优化版"""
    print("🔬 层次化损失函数 + 中文预处理对比实验 (优化版)")
    print("🚀 性能优化：增大batch_size, 减少epoch, 数据缓存, 激进早停")
    print("=" * 80)
    
    if not HIERARCHICAL_AVAILABLE:
        print("❌ 层次化功能不可用")
        return
    
    csv_path = "lunwenimpro/newjob1_sortall.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        return
    
    print("选择实验规模 (优化版):")
    print("1. 快速测试 (4K样本, 3epoch) - 推荐用于调试")
    print("2. 标准测试 (6K样本, 3epoch)")
    print("3. 完整测试 (8K样本, 3epoch)")
    
    choice = input("请输入选择 (1-3): ")
    
    if choice == "1":
        max_samples = 4000
    elif choice == "2":
        max_samples = 6000
    elif choice == "3":
        max_samples = 8000
    else:
        print("无效选择,使用快速测试")
        max_samples = 4000
    
    print(f"\n🎯 优化版实验配置:")
    print(f"   数据文件: {csv_path}")
    print(f"   样本限制: {max_samples}")
    print(f"   批次大小: 64 (优化后)")
    print(f"   最大epoch: 3 (优化后)")
    print(f"   早停耐心: 2 (优化后)")
    print(f"   测试模型: 仅chinese-roberta-wwm-ext (优化后)")
    print(f"   数据缓存: 启用 (优化后)")
    print(f"   预计单个实验时间: 300-600秒")
    if len(['hfl/chinese-roberta-wwm-ext']) == 1:  # 如果只测试一个模型
        print(f"   测试模型: hfl/chinese-roberta-wwm-ext (优化后)")
    else:  # 如果测试多个模型
        print(f"   测试模型: {len(['bert-base-chinese', 'hfl/chinese-bert-wwm-ext', 'hfl/chinese-roberta-wwm-ext'])} 个中文预训练模型")
    
    print(f"   数据缓存: 启用 (优化后)")
    print(f"   预计单个实验时间: 300-600秒")
    
    confirm = input(f"\n确认开始优化版实验? (y/N): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return
    
    try:
        # 创建优化版实验运行器
        runner = HierarchicalExperimentRunner(csv_path, max_samples)
        
        # 运行对比实验
        results, results_dir = runner.run_hierarchical_comparison()
        
        print(f"\n🎉 优化版实验完成!")
        print(f"📁 查看详细结果: {results_dir}")
        
    except Exception as e:
        print(f"❌ 实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()