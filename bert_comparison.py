#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新的中文BERT模型对比脚本
移除TensorFlow依赖，添加更多PyTorch原生模型
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

# Windows多进程修复
import torch
torch.set_float32_matmul_precision('medium')

if __name__ == '__main__':
    from job_offers_classifier.job_offers_classfier_old import (
        ChineseTransformerJobOffersClassifier,
        get_recommended_chinese_models
    )
    from job_offers_classifier.job_offers_utils_old import create_hierarchy_node
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score

    # 更新的模型配置 - 只包含PyTorch原生支持的模型
    UPDATED_MODELS = {
        'google_bert': {
            'name': 'bert-base-chinese',
            'description': 'Google Chinese BERT-base',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': 'Google官方中文BERT'
        },
        'ckiplab_bert': {
            'name': 'ckiplab/bert-base-chinese',
            'description': 'CKIP Chinese BERT-base',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': '台湾中研院中文BERT'
        },
        'hfl_roberta': {
            'name': 'hfl/chinese-roberta-wwm-ext',
            'description': 'HFL Chinese RoBERTa-wwm-ext (推荐)',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': '哈工大RoBERTa，性能最佳'
        },
        'hfl_bert': {
            'name': 'hfl/chinese-bert-wwm-ext',
            'description': 'HFL Chinese BERT-wwm-ext',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': '哈工大BERT，经典稳定'
        },
        'hfl_electra': {
            'name': 'hfl/chinese-electra-180g-base-discriminator',
            'description': 'HFL Chinese ELECTRA-base',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': '哈工大ELECTRA，高效训练'
        },
        'bert_base_multilingual': {
            'name': 'bert-base-multilingual-cased',
            'description': 'BERT Multilingual (包含中文)',
            'learning_rate': 2e-5,
            'batch_size': 16,
            'notes': '多语言BERT，作为基准对比'
        }
    }

    def setup_comparison_environment():
        """设置对比环境"""
        print("🔧 设置模型对比环境...")
        
        results_dir = Path("chinese_bert_comparison_updated")
        results_dir.mkdir(exist_ok=True)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU: {gpu_name}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   PyTorch版本: {torch.__version__}")
        else:
            print("⚠️  未检测到GPU，将使用CPU训练")
        
        return results_dir

    def load_and_prepare_data(csv_path, test_size=0.2, max_samples=None):
        """加载和准备数据"""
        print(f"📊 加载数据: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        print(f"   原始数据: {len(df)} 行")
        
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
            print(f"   随机采样: {len(df)} 行")
        
        # 组合文本特征
        def combine_features(row):
            parts = []
            for col in ['岗位', '岗位描述', '岗位职能']:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
            return ' '.join(parts)
        
        df['combined_text'] = df.apply(combine_features, axis=1)
        df['isco_code'] = df['ISCO_4_Digit_Code_Gemini'].astype(str).str.zfill(4)
        
        # 数据统计
        isco_counts = df['isco_code'].value_counts()
        print(f"   唯一ISCO编码: {len(isco_counts)}")
        print(f"   平均文本长度: {df['combined_text'].str.len().mean():.0f} 字符")
        
        # 智能数据划分
        rare_threshold = 3
        rare_classes = isco_counts[isco_counts <= rare_threshold].index.tolist()
        common_classes = isco_counts[isco_counts > rare_threshold].index.tolist()
        
        print(f"   稀有类别 (≤{rare_threshold}样本): {len(rare_classes)}")
        print(f"   常见类别 (>{rare_threshold}样本): {len(common_classes)}")
        
        # 分别处理稀有和常见类别
        train_indices = []
        test_indices = []
        
        for rare_class in rare_classes:
            rare_samples = df[df['isco_code'] == rare_class].index.tolist()
            if len(rare_samples) == 1:
                train_indices.extend(rare_samples)
            elif len(rare_samples) == 2:
                train_indices.append(rare_samples[0])
                test_indices.append(rare_samples[1])
            else:
                train_indices.extend(rare_samples[:2])
                test_indices.extend(rare_samples[2:])
        
        if common_classes:
            common_df = df[df['isco_code'].isin(common_classes)]
            train_common, test_common = train_test_split(
                common_df, test_size=test_size, random_state=42,
                stratify=common_df['isco_code']
            )
            train_indices.extend(train_common.index.tolist())
            test_indices.extend(test_common.index.tolist())
        
        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]
        
        print(f"✅ 数据划分完成:")
        print(f"   训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"   测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        # 创建层次结构
        hierarchy = create_isco_hierarchy_from_codes(df['isco_code'].unique())
        
        return (train_df['combined_text'].tolist(),
                train_df['isco_code'].tolist(),
                test_df['combined_text'].tolist(), 
                test_df['isco_code'].tolist(),
                hierarchy,
                {
                    'total_samples': len(df),
                    'train_samples': len(train_df),
                    'test_samples': len(test_df),
                    'num_classes': len(set(df['isco_code'])),
                    'rare_classes': len(rare_classes),
                    'common_classes': len(common_classes)
                })

    def create_isco_hierarchy_from_codes(isco_codes):
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

    def test_model_availability(model_config):
        """测试模型是否可用"""
        model_name = model_config['name']
        print(f"   🔍 测试模型: {model_name}")
        
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            
            config = AutoConfig.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 尝试加载模型来测试
            model = AutoModel.from_pretrained(model_name)
            
            # 清理内存
            del model
            del tokenizer
            
            print(f"      ✅ 模型可用")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "TensorFlow" in error_msg or "from_tf=True" in error_msg:
                print(f"      ❌ 模型需要TensorFlow权重转换")
            else:
                print(f"      ❌ 模型不可用: {e}")
            return False

    def train_single_model(model_key, model_config, train_texts, train_labels, 
                          test_texts, test_labels, hierarchy, results_dir, 
                          training_config):
        """训练单个模型"""
        print(f"\n🤖 开始训练: {model_config['description']}")
        print(f"   模型: {model_config['name']}")
        print(f"   说明: {model_config['notes']}")
        
        # 测试模型可用性
        if not test_model_availability(model_config):
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
                'error': 'Model not available (possibly requires TensorFlow)'
            }
        
        start_time = time.time()
        model_dir = results_dir / f"model_{model_key}"
        
        try:
            # 创建分类器
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
                verbose=False
            )
            
            # 准备验证集
            val_size = min(1000, len(test_texts) // 3)
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
            detailed_results.to_csv(results_dir / f"{model_key}_detailed_results.csv", 
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
                'status': 'success'
            }
            
            print(f"✅ 训练完成!")
            print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Top-3准确率: {top_3_acc:.4f} ({top_3_acc*100:.2f}%)")
            print(f"   Top-5准确率: {top_5_acc:.4f} ({top_5_acc*100:.2f}%)")
            print(f"   训练时间: {training_time/60:.1f} 分钟")
            
            return result
            
        except Exception as e:
            print(f"❌ 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'model_key': model_key,
                'model_name': model_config['name'],
                'description': model_config['description'],
                'notes': model_config['notes'],
                'accuracy': 0.0,
                'top_3_accuracy': 0.0,
                'top_5_accuracy': 0.0,
                'training_time_minutes': 0.0,
                'test_samples': 0,
                'status': 'failed',
                'error': str(e)
            }

    def run_model_comparison(csv_path, models_to_test=None, max_samples=None):
        """运行模型对比"""
        print("🚀 开始更新版中文BERT模型对比评估")
        print("🔧 已移除TensorFlow依赖模型，添加新的PyTorch模型")
        print("=" * 60)
        
        results_dir = setup_comparison_environment()
        
        train_texts, train_labels, test_texts, test_labels, hierarchy, data_info = load_and_prepare_data(
            csv_path, max_samples=max_samples)
        
        # 训练配置
        training_config = {
            'max_epochs': 3,
            'patience': 2,
            'max_seq_length': 256
        }
        
        if models_to_test is None:
            models_to_test = list(UPDATED_MODELS.keys())
        
        print(f"\n📋 将测试以下模型:")
        for model_key in models_to_test:
            model_config = UPDATED_MODELS[model_key]
            print(f"   {model_key}: {model_config['description']}")
            print(f"      说明: {model_config['notes']}")
        
        total_start_time = time.time()
        results = []
        
        for i, model_key in enumerate(models_to_test, 1):
            print(f"\n{'='*60}")
            print(f"📊 进度: {i}/{len(models_to_test)} - {model_key}")
            print(f"{'='*60}")
            
            model_config = UPDATED_MODELS[model_key]
            result = train_single_model(
                model_key, model_config, train_texts, train_labels,
                test_texts, test_labels, hierarchy, results_dir, training_config
            )
            results.append(result)
            
            # 保存中间结果
            interim_df = pd.DataFrame(results)
            interim_df.to_csv(results_dir / "interim_results.csv", index=False, encoding='utf-8')
        
        total_time = time.time() - total_start_time
        generate_comparison_report(results, data_info, total_time, results_dir)
        
        return results, results_dir

    def generate_comparison_report(results, data_info, total_time, results_dir):
        """生成对比报告"""
        print(f"\n📊 生成对比报告...")
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        results_df.to_csv(results_dir / "updated_comparison_results.csv", index=False, encoding='utf-8')
        
        report = {
            'experiment_info': {
                'experiment_type': 'Updated Chinese BERT Model Comparison (No TensorFlow)',
                'timestamp': datetime.now().isoformat(),
                'total_time_hours': total_time / 3600,
                'data_info': data_info,
                'python_version': f"Python {'.'.join(map(str, __import__('sys').version_info[:3]))}",
                'pytorch_version': torch.__version__
            },
            'results': results_df.to_dict('records'),
            'summary': {
                'best_model': results_df.iloc[0]['model_name'] if len(results_df) > 0 else None,
                'best_accuracy': results_df.iloc[0]['accuracy'] if len(results_df) > 0 else 0,
                'models_tested': len(results_df),
                'successful_models': len(results_df[results_df['status'] == 'success']),
                'failed_models': len(results_df[results_df['status'] == 'failed'])
            }
        }
        
        with open(results_dir / "updated_comparison_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 更新版模型对比完成!")
        print(f"📈 结果总结:")
        print(f"   总耗时: {total_time/3600:.2f} 小时")
        print(f"   成功模型: {report['summary']['successful_models']}/{report['summary']['models_tested']}")
        print(f"   失败模型: {report['summary']['failed_models']}")
        
        if len(results_df) > 0:
            successful_results = results_df[results_df['status'] == 'success']
            
            if len(successful_results) > 0:
                print(f"\n🏆 模型排名 (按准确率):")
                for i, (_, row) in enumerate(successful_results.iterrows(), 1):
                    print(f"   {i}. {row['description']}")
                    print(f"      模型: {row['model_name']}")
                    print(f"      准确率: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)")
                    print(f"      Top-3: {row['top_3_accuracy']:.4f} ({row['top_3_accuracy']*100:.2f}%)")
                    print(f"      Top-5: {row['top_5_accuracy']:.4f} ({row['top_5_accuracy']*100:.2f}%)")
                    print(f"      训练时间: {row['training_time_minutes']:.1f} 分钟")
                    print(f"      说明: {row['notes']}")
                    print()
            
            failed_results = results_df[results_df['status'] == 'failed']
            if len(failed_results) > 0:
                print(f"❌ 失败模型:")
                for _, row in failed_results.iterrows():
                    print(f"   - {row['description']}: {row.get('error', 'Unknown error')}")
        
        print(f"\n📁 详细结果保存在: {results_dir}")

    def main():
        """主函数"""
        print("🚀 更新版中文BERT模型对比评估")
        print("🔧 移除TensorFlow依赖，支持Python 3.13")
        print("=" * 60)
        
        print("选择测试模式:")
        print("1. 完整测试 (所有6个模型)")
        print("2. 快速测试 (限制样本数)")
        print("3. 优质模型测试 (只测试哈工大3个模型)")
        print("4. 基准对比测试 (Google + HFL RoBERTa)")
        
        choice = input("请输入选择 (1-4): ")
        
        if choice == "1":
            models_to_test = list(UPDATED_MODELS.keys())
            max_samples = None
        elif choice == "2":
            models_to_test = list(UPDATED_MODELS.keys())
            max_samples = 15000
        elif choice == "3":
            models_to_test = ['hfl_roberta', 'hfl_bert', 'hfl_electra']
            max_samples = 15000
        elif choice == "4":
            models_to_test = ['google_bert', 'hfl_roberta']
            max_samples = 10000
        else:
            print("无效选择，使用快速测试")
            models_to_test = list(UPDATED_MODELS.keys())
            max_samples = 15000
        
        csv_path = "newjob1_sortall.csv"
        if not os.path.exists(csv_path):
            print(f"❌ 找不到数据文件: {csv_path}")
            return
        
        print(f"\n🎯 即将测试的模型:")
        for model_key in models_to_test:
            config = UPDATED_MODELS[model_key]
            print(f"   - {config['description']}")
            print(f"     {config['notes']}")
        
        if max_samples:
            print(f"\n⚡ 快速模式: 限制样本数为 {max_samples:,}")
        
        confirm = input(f"\n确认开始测试? (y/N): ")
        if confirm.lower() != 'y':
            print("测试已取消")
            return
        
        results, results_dir = run_model_comparison(csv_path, models_to_test, max_samples)
        
        print(f"\n🎯 更新版对比完成! 查看详细结果: {results_dir}")

    main()