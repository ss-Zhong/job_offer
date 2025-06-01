#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版中文BERT训练流程 - 集成数据增强与预处理
基于原有代码，集成新的数据增强功能
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
    
    # 导入新的数据增强模块
    from chinese_job_data_augmentation import EnhancedJobDataProcessor


class EnhancedModelTrainer:
    """增强版模型训练器"""
    
    def __init__(self, 
                 data_processor: EnhancedJobDataProcessor,
                 enable_augmentation: bool = True,
                 balance_data: bool = True,
                 target_samples_per_class: int = 8):
        
        self.data_processor = data_processor
        self.enable_augmentation = enable_augmentation
        self.balance_data = balance_data
        self.target_samples_per_class = target_samples_per_class
        
        # 更新的模型配置
        self.ENHANCED_MODELS = {
            'hfl_roberta_enhanced': {
                'name': 'hfl/chinese-roberta-wwm-ext',
                'description': 'HFL Chinese RoBERTa-wwm-ext (数据增强版)',
                'learning_rate': 2e-5,
                'batch_size': 16,
                'notes': '哈工大RoBERTa + 数据增强，性能最佳'
            },
            'hfl_bert_enhanced': {
                'name': 'hfl/chinese-bert-wwm-ext',
                'description': 'HFL Chinese BERT-wwm-ext (数据增强版)',
                'learning_rate': 2e-5,
                'batch_size': 16,
                'notes': '哈工大BERT + 数据增强，稳定可靠'
            },
            'google_bert_enhanced': {
                'name': 'bert-base-chinese',
                'description': 'Google Chinese BERT (数据增强版)',
                'learning_rate': 2e-5,
                'batch_size': 16,
                'notes': 'Google中文BERT + 数据增强'
            }
        }
    
    def load_and_prepare_enhanced_data(self, csv_path, test_size=0.2, max_samples=None):
        """加载和准备增强数据"""
        print(f"📊 加载和准备增强数据: {csv_path}")
        
        # 使用增强处理器处理数据
        all_texts, all_labels, processing_stats = self.data_processor.process_csv_data(
            csv_path=csv_path,
            enable_augmentation=self.enable_augmentation,
            balance_data=self.balance_data,
            target_samples_per_class=self.target_samples_per_class
        )
        
        # 如果设置了最大样本数限制
        if max_samples and len(all_texts) > max_samples:
            # 分层采样保持类别平衡
            from sklearn.model_selection import train_test_split
            all_texts, _, all_labels, _ = train_test_split(
                all_texts, all_labels, 
                train_size=max_samples,
                stratify=all_labels,
                random_state=42
            )
            print(f"   限制样本数: {len(all_texts)}")
        
        # 划分训练测试集
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            all_texts, all_labels,
            test_size=test_size,
            random_state=42,
            stratify=all_labels
        )
        
        # 创建层次结构
        hierarchy = self.create_isco_hierarchy_from_codes(set(all_labels))
        
        # 数据统计
        data_info = {
            'total_samples': len(all_texts),
            'train_samples': len(train_texts),
            'test_samples': len(test_texts),
            'num_classes': len(set(all_labels)),
            'augmentation_enabled': self.enable_augmentation,
            'balance_enabled': self.balance_data,
            'original_stats': processing_stats['original_stats'],
            'final_stats': processing_stats['final_stats']
        }
        
        print(f"✅ 增强数据准备完成:")
        print(f"   原始样本: {processing_stats['original_stats']['total_samples']}")
        print(f"   增强后样本: {processing_stats['final_stats']['total_samples']}")
        print(f"   训练集: {len(train_texts)}")
        print(f"   测试集: {len(test_texts)}")
        print(f"   类别数: {len(set(all_labels))}")
        
        return (train_texts, train_labels, test_texts, test_labels, hierarchy, data_info)
    
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
    
    def train_enhanced_model(self, model_key, model_config, train_texts, train_labels, 
                           test_texts, test_labels, hierarchy, results_dir, 
                           training_config):
        """训练增强模型"""
        print(f"\n🤖 开始训练增强模型: {model_config['description']}")
        print(f"   模型: {model_config['name']}")
        print(f"   数据增强: {'✓' if self.enable_augmentation else '✗'}")
        print(f"   数据平衡: {'✓' if self.balance_data else '✗'}")
        
        start_time = time.time()
        model_dir = results_dir / f"enhanced_model_{model_key}"
        
        try:
            # 创建分类器（使用增强配置）
            classifier = ChineseTransformerJobOffersClassifier(
                model_dir=str(model_dir),
                hierarchy=hierarchy,
                transformer_model=model_config['name'],
                learning_rate=model_config['learning_rate'],
                batch_size=model_config['batch_size'],
                max_epochs=training_config['max_epochs'],
                early_stopping=True,
                early_stopping_patience=training_config['patience'],
                max_sequence_length=training_config['max_seq_length'],
                devices=1,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                precision="16-mixed" if torch.cuda.is_available() else 32,
                threads=0,
                verbose=True
            )
            
            # 准备验证集
            val_size = min(500, len(test_texts) // 3)
            val_texts = test_texts[:val_size]
            val_labels = test_labels[:val_size]
            final_test_texts = test_texts[val_size:]
            final_test_labels = test_labels[val_size:]
            
            print(f"   训练样本: {len(train_texts):,}")
            print(f"   验证样本: {val_size:,}")
            print(f"   测试样本: {len(final_test_texts):,}")
            
            # 训练模型
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
            
            # 保存详细结果
            detailed_results = pd.DataFrame({
                'true_label': y_true,
                'predicted_label': y_pred,
                'confidence': predictions_df['prob_1'].tolist(),
                'correct': [t == p for t, p in zip(y_true, y_pred)]
            })
            detailed_results.to_csv(results_dir / f"enhanced_{model_key}_detailed_results.csv", 
                                  index=False, encoding='utf-8')
            
            result = {
                'model_key': model_key,
                'model_name': model_config['name'],
                'description': model_config['description'],
                'notes': model_config['notes'],
                'accuracy': accuracy,
                'top_3_accuracy': top_3_acc,
                'top_5_accuracy': top_5_acc,
                'training_time_minutes': training_time / 60,
                'test_samples': len(final_test_labels),
                'status': 'success',
                'enhancement_info': {
                    'augmentation_enabled': self.enable_augmentation,
                    'balance_enabled': self.balance_data,
                    'target_samples_per_class': self.target_samples_per_class
                }
            }
            
            print(f"✅ 增强模型训练完成!")
            print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Top-3准确率: {top_3_acc:.4f} ({top_3_acc*100:.2f}%)")
            print(f"   Top-5准确率: {top_5_acc:.4f} ({top_5_acc*100:.2f}%)")
            print(f"   训练时间: {training_time/60:.1f} 分钟")
            
            return result
            
        except Exception as e:
            print(f"❌ 增强模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_key': model_key,
                'model_name': model_config['name'],
                'description': model_config['description'],
                'accuracy': 0.0,
                'top_3_accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'training_time_minutes': 0.0,
                'test_samples': 0,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_enhanced_comparison(self, csv_path, models_to_test=None, max_samples=None):
        """运行增强版模型对比"""
        print("🚀 开始增强版中文BERT模型对比评估")
        print("🎯 集成数据增强与预处理功能")
        print("=" * 60)
        
        # 创建结果目录
        results_dir = Path("enhanced_chinese_bert_comparison")
        results_dir.mkdir(exist_ok=True)
        
        # 准备增强数据
        train_texts, train_labels, test_texts, test_labels, hierarchy, data_info = \
            self.load_and_prepare_enhanced_data(csv_path, max_samples=max_samples)
        
        # 训练配置
        training_config = {
            'max_epochs': 5,  # 增加epoch数以适应增强数据
            'patience': 3,
            'max_seq_length': 256
        }
        
        if models_to_test is None:
            models_to_test = list(self.ENHANCED_MODELS.keys())
        
        print(f"\n📋 将测试以下增强模型:")
        for model_key in models_to_test:
            model_config = self.ENHANCED_MODELS[model_key]
            print(f"   {model_key}: {model_config['description']}")
            print(f"      说明: {model_config['notes']}")
        
        total_start_time = time.time()
        results = []
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n{'='*60}")
            print(f"📊 进度: {i}/{len(models_to_test)} - {model_key}")
            print(f"{'='*60}")
            
            model_config = self.ENHANCED_MODELS[model_key]
            result = self.train_enhanced_model(
                model_key, model_config, train_texts, train_labels,
                test_texts, test_labels, hierarchy, results_dir, training_config
            )
            results.append(result)
            
            # 保存中间结果
            interim_df = pd.DataFrame(results)
            interim_df.to_csv(results_dir / "enhanced_interim_results.csv", index=False, encoding='utf-8')
        
        total_time = time.time() - total_start_time
        self.generate_enhanced_comparison_report(results, data_info, total_time, results_dir)
        
        return results, results_dir
    
    def generate_enhanced_comparison_report(self, results, data_info, total_time, results_dir):
        """生成增强版对比报告"""
        print(f"\n📊 生成增强版对比报告...")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        results_df.to_csv(results_dir / "enhanced_comparison_results.csv", index=False, encoding='utf-8')
        
        report = {
            'experiment_info': {
                'experiment_type': 'Enhanced Chinese BERT Model Comparison with Data Augmentation',
                'timestamp': datetime.now().isoformat(),
                'total_time_hours': total_time / 3600,
                'data_info': data_info,
                'enhancement_settings': {
                    'augmentation_enabled': self.enable_augmentation,
                    'balance_enabled': self.balance_data,
                    'target_samples_per_class': self.target_samples_per_class
                },
                'python_version': f"Python {'.'.join(map(str, __import__('sys').version_info[:3]))}",
                'pytorch_version': torch.__version__
            },
            'results': results_df.to_dict('records'),
            'summary': {
                'best_model': results_df.iloc[0]['model_name'] if len(results_df) > 0 else None,
                'best_accuracy': results_df.iloc[0]['accuracy'] if len(results_df) > 0 else 0,
                'models_tested': len(results_df),
                'successful_models': len(results_df[results_df['status'] == 'success']),
                'failed_models': len(results_df[results_df['status'] == 'failed']),
                'avg_improvement': self._calculate_improvement(results_df)
            }
        }
        
        with open(results_dir / "enhanced_comparison_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 增强版模型对比完成!")
        print(f"📈 结果总结:")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        print(f"   成功模型: {report['summary']['successful_models']}/{report['summary']['models_tested']}")
        print(f"   数据增强: {'✓' if self.enable_augmentation else '✗'}")
        print(f"   数据平衡: {'✓' if self.balance_data else '✗'}")
        
        if len(results_df) > 0:
            successful_results = results_df[results_df['status'] == 'success']
            
            if len(successful_results) > 0:
                print(f"\n🏆 增强模型排名 (按准确率):")
                for i, (_, row) in enumerate(successful_results.iterrows(), 1):
                    print(f"   {i}. {row['description']}")
                    print(f"      准确率: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)")
                    print(f"      Top-3: {row['top_3_accuracy']:.4f} ({row['top_3_accuracy']*100:.2f}%)")
                    print(f"      训练时间: {row['training_time_minutes']:.1f} 分钟")
                    print(f"      增强说明: {row['notes']}")
                    print()
        
        print(f"\n📁 详细结果保存在: {results_dir}")
    
    def _calculate_improvement(self, results_df):
        """计算相比基准模型的改进"""
        # 这里可以添加与原始模型对比的逻辑
        # 暂时返回平均准确率
        successful = results_df[results_df['status'] == 'success']
        if len(successful) > 0:
            return successful['accuracy'].mean()
        return 0.0


def main():
    """主函数 - 增强版训练流程"""
    print("🚀 增强版中文BERT模型训练系统")
    print("🎯 集成数据增强与预处理功能")
    print("=" * 60)
    
    # 创建数据处理器
    print("🔧 初始化数据处理器...")
    data_processor = EnhancedJobDataProcessor(
        job_synonyms_path="job_synonyms.json"  # 如果有的话
    )
    
    # 创建增强训练器
    trainer = EnhancedModelTrainer(
        data_processor=data_processor,
        enable_augmentation=True,      # 启用数据增强
        balance_data=True,            # 启用数据平衡
        target_samples_per_class=8    # 每类目标样本数
    )
    
    print("选择测试模式:")
    print("1. 增强版完整测试 (推荐)")
    print("2. 增强版快速测试")
    print("3. 数据增强效果对比")
    print("4. 只测试最佳模型")
    
    choice = input("请输入选择 (1-4): ")
    
    if choice == "1":
        models_to_test = list(trainer.ENHANCED_MODELS.keys())
        max_samples = None
    elif choice == "2":
        models_to_test = list(trainer.ENHANCED_MODELS.keys())
        max_samples = 10000
    elif choice == "3":
        # 对比增强前后效果
        models_to_test = ['hfl_roberta_enhanced']
        max_samples = 8000
    elif choice == "4":
        models_to_test = ['hfl_roberta_enhanced']
        max_samples = 15000
    else:
        print("无效选择，使用推荐配置")
        models_to_test = list(trainer.ENHANCED_MODELS.keys())
        max_samples = 15000
    
    csv_path = "newjob1_sortall.csv"
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}")
        print("请确保数据文件在当前目录下")
        return
    
    print(f"\n🎯 即将运行增强版训练:")
    print(f"   数据文件: {csv_path}")
    print(f"   数据增强: ✓")
    print(f"   数据平衡: ✓")
    print(f"   测试模型: {len(models_to_test)} 个")
    if max_samples:
        print(f"   样本限制: {max_samples:,}")
    
    confirm = input(f"\n确认开始训练? (y/N): ")
    if confirm.lower() != 'y':
        print("训练已取消")
        return
    
    try:
        # 运行增强版对比
        results, results_dir = trainer.run_enhanced_comparison(
            csv_path, models_to_test, max_samples
        )
        
        print(f"\n🎯 增强版训练完成! 查看详细结果: {results_dir}")
        
        # 显示最佳结果
        if results:
            best_result = max(results, key=lambda x: x.get('accuracy', 0))
            if best_result['status'] == 'success':
                print(f"\n🏆 最佳模型表现:")
                print(f"   模型: {best_result['description']}")
                print(f"   准确率: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
                print(f"   Top-3准确率: {best_result['top_3_accuracy']:.4f}")
                print(f"   Top-5准确率: {best_result['top_5_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()