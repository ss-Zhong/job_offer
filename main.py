#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全取代main.py的中文职业分类训练脚本
支持真实数据训练 + 命令行接口
"""

import os
from datetime import datetime
import click
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Windows多进程修复
torch.set_float32_matmul_precision('medium')

from job_offers_classifier.datasets import *
from job_offers_classifier.job_offers_classfier import *
from job_offers_classifier.job_offers_utils import *
from job_offers_classifier.load_save import *


@click.command()
# Command
@click.argument("command", type=str)
@click.argument("classifier", type=str)
# General settings
@click.option("-x", "--x_data", type=str, required=True)
@click.option("-y", "--y_data", type=str, required=False, default="")
@click.option("-h", "--hierarchy_data", type=str, required=False, default="")
@click.option("-m", "--model_dir", type=str, required=True, default="model")
# Transformer model settings
@click.option("-t", "--transformer_model", type=str, required=False, default="hfl/chinese-roberta-wwm-ext")
@click.option("-tc", "--transformer_ckpt_path", type=str, required=False, default="")
@click.option("-mm", "--modeling_mode", type=str, required=False, default="bottom-up")
# Training parameters
@click.option("-l", "--learning_rate", type=float, required=False, default=2e-5)
@click.option("-w", "--weight_decay", type=float, required=False, default=0.01)
@click.option("-e", "--max_epochs", type=int, required=False, default=10)
@click.option("-b", "--batch_size", type=int, required=False, default=16)
@click.option("-s", "--max_sequence_length", type=int, required=False, default=256)
# Early stopping
@click.option("--early_stopping", type=bool, required=False, default=True)
@click.option("--early_stopping_delta", type=float, required=False, default=0.001)
@click.option("--early_stopping_patience", type=int, required=False, default=3)
# Hardware
@click.option("-T", "--threads", type=int, required=False, default=0)  # 默认0避免多进程问题
@click.option("-D", "--devices", type=int, required=False, default=1)
@click.option("-P", "--precision", type=str, required=False, default="16-mixed")  # 改为字符串支持16-mixed
@click.option("-A", "--accelerator", type=str, required=False, default="auto")
# Linear model
@click.option("--eps", type=float, required=False, default=0.001)
@click.option("-c", "--cost", type=float, required=False, default=10)
@click.option("-E", "--ensemble", type=int, required=False, default=1)
@click.option("--use_provided_hierarchy", type=int, required=False, default=1)
@click.option("--tfidf_vectorizer_min_df", type=int, required=False, default=2)
# 新增功能选项
@click.option("--language", type=str, required=False, default="zh", help="Language for text processing")
@click.option("--csv_mode", type=bool, required=False, default=False, help="Use CSV file with auto split")
@click.option("--test_size", type=float, required=False, default=0.2, help="Test set ratio for CSV mode")
@click.option("--val_texts", type=str, required=False, default="", help="Validation texts file")
@click.option("--val_labels", type=str, required=False, default="", help="Validation labels file")
# Prediction
@click.option("-p", "--pred_path", type=str, required=False, default="")
@click.option("-S", "--seed", type=int, required=False, default=1993)
@click.option("-v", "--verbose", type=bool, required=False, default=True)
def main(command: str,
         classifier: str,
         x_data: str,
         y_data: str,
         hierarchy_data: str,
         model_dir: str,

         transformer_model: str,
         transformer_ckpt_path: str,
         modeling_mode: str,

         learning_rate: float,
         weight_decay: float,
         max_epochs: int,
         batch_size: int,
         max_sequence_length: int,

         early_stopping: bool,
         early_stopping_delta: float,
         early_stopping_patience: int,

         threads: int,
         devices: int,
         precision: str,  # 改为字符串
         accelerator: str,

         eps: float,
         cost: float,
         ensemble: int,
         use_provided_hierarchy: int,
         tfidf_vectorizer_min_df: int,

         language: str,
         csv_mode: bool,
         test_size: float,
         val_texts: str,
         val_labels: str,
         pred_path: str,
         seed: int,
         verbose: bool,
         ):

    # 设置线程数
    if threads <= 0:
        threads = 0  # 禁用多进程
    
    # GPU设备检测
    if torch.cuda.is_available():
        devices = min(devices, torch.cuda.device_count())
        if accelerator == "auto":
            accelerator = "gpu"
    else:
        if accelerator == "auto":
            accelerator = "cpu"
        devices = 1
        
    print(f"🚀 Starting {command} with {classifier}")
    print(f"⏰ Time: {datetime.now()}")
    print(f"🌐 Language: {language}")
    print(f"🤖 Model: {transformer_model}")
    print(f"💻 Device: {accelerator.upper()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    if command == 'fit':
        if csv_mode or x_data.endswith('.csv'):
            # CSV模式：自动处理CSV文件
            print(f"\n📊 CSV模式：处理 {x_data}")
            X, y, hierarchy, X_val, y_val = process_csv_file(x_data, hierarchy_data, test_size, seed, verbose)
        else:
            # 传统模式：分别加载文件
            print(f"\n📁 传统模式：分别加载文件")
            if not hierarchy_data:
                raise ValueError("传统模式需要提供hierarchy_data")
            
            hierarchy_df = load_to_df(hierarchy_data)
            hierarchy = create_hierarchy(hierarchy_df)
            
            X = load_texts(x_data)
            y = load_texts(y_data)
            
            # 加载验证集（如果提供）
            X_val = None
            y_val = None
            if val_texts and val_labels:
                X_val = load_texts(val_texts)
                y_val = load_texts(val_labels)

        # 创建模型
        if classifier == "ChineseLinearJobOffersClassifier" or classifier == "LinearJobOffersClassifier":
            model = ChineseLinearJobOffersClassifier(
                model_dir=model_dir,
                hierarchy=hierarchy,
                eps=eps,
                c=cost,
                use_provided_hierarchy=use_provided_hierarchy,
                ensemble=ensemble,
                threads=threads,
                tfidf_vectorizer_min_df=tfidf_vectorizer_min_df,
                verbose=verbose
            )
        elif classifier == "ChineseTransformerJobOffersClassifier" or classifier == "TransformerJobOffersClassifier":
            # 验证中文模型
            if language == 'zh' and 'chinese' not in transformer_model.lower() and 'hfl' not in transformer_model.lower():
                print(f"⚠️  Warning: Using '{transformer_model}' for Chinese text")
                print(f"💡 Consider using 'hfl/chinese-roberta-wwm-ext'")
            
            # 处理precision参数
            if precision.isdigit():
                precision = int(precision)
            
            model = ChineseTransformerJobOffersClassifier(
                model_dir=model_dir,
                hierarchy=hierarchy,
                transformer_model=transformer_model,
                transformer_ckpt_path=transformer_ckpt_path,
                modeling_mode=modeling_mode,
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
                threads=threads,
                precision=precision,
                verbose=verbose
            )
        else:
            raise ValueError(f'Unknown classifier type: {classifier}')

        # 训练模型
        print(f"\n🎯 开始训练...")
        print(f"   训练样本: {len(X):,}")
        if X_val:
            print(f"   验证样本: {len(X_val):,}")
        print(f"   类别数量: {len(set(y))}")
        
        model.fit(y, X, y_val=y_val, X_val=X_val)
        print(f"✅ 训练完成！")

    elif command == 'predict':
        # 预测模式
        X = load_texts(x_data)
        
        if classifier == "ChineseLinearJobOffersClassifier" or classifier == "LinearJobOffersClassifier":
            model = ChineseLinearJobOffersClassifier(threads=threads)
        elif classifier == "ChineseTransformerJobOffersClassifier" or classifier == "TransformerJobOffersClassifier":
            # 处理precision参数
            if precision.isdigit():
                precision = int(precision)
                
            model = ChineseTransformerJobOffersClassifier(
                batch_size=batch_size,
                devices=devices,
                threads=threads,
                precision=precision,
                accelerator=accelerator,
            )
        else:
            raise ValueError(f'Unknown classifier type: {classifier}')
            
        print(f"\n🔮 加载模型并预测...")
        model.load(model_dir)
        pred, pred_map = model.predict(X)
        
        # 保存预测结果
        np.savetxt(pred_path, pred)
        save_as_text(f"{pred_path}.map", list(pred_map.values()))
        print(f"✅ 预测完成！结果保存到: {pred_path}")
        
    else:
        raise ValueError(f'Unknown command: {command}')

    print(f"\n🎉 All done! Time: {datetime.now()}")


def process_csv_file(csv_path, hierarchy_path, test_size, seed, verbose):
    """处理CSV文件，自动划分训练测试集"""
    print(f"📊 处理CSV文件: {csv_path}")
    
    # 加载CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    
    print(f"   总样本数: {len(df)}")
    
    # 检查必需的列
    required_cols = ['岗位', '岗位描述', '岗位职能', 'ISCO_4_Digit_Code_Gemini']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必需的列: {missing_cols}")
        print(f"   当前列: {list(df.columns)}")
        raise ValueError(f"CSV文件缺少必需的列: {missing_cols}")
    
    # 组合文本特征
    def combine_features(row):
        parts = []
        for col in ['岗位', '岗位描述', '岗位职能']:
            if pd.notna(row[col]):
                parts.append(str(row[col]))
        return ' '.join(parts)
    
    df['combined_text'] = df.apply(combine_features, axis=1)
    
    # 处理ISCO编码
    df['isco_code'] = df['ISCO_4_Digit_Code_Gemini'].astype(str).str.zfill(4)
    
    # 统计信息
    isco_counts = df['isco_code'].value_counts()
    print(f"   唯一ISCO编码: {len(isco_counts)}")
    print(f"   最频繁的5个编码: {isco_counts.head().to_dict()}")
    
    # 智能划分：确保训练集包含所有类别
    rare_threshold = 2
    rare_classes = isco_counts[isco_counts <= rare_threshold].index.tolist()
    common_classes = isco_counts[isco_counts > rare_threshold].index.tolist()
    
    print(f"   稀有类别 (≤{rare_threshold}样本): {len(rare_classes)}")
    print(f"   常见类别 (>{rare_threshold}样本): {len(common_classes)}")
    
    # 分别处理稀有和常见类别
    train_indices = []
    test_indices = []
    
    # 稀有类别：至少1个在训练集
    for rare_class in rare_classes:
        rare_samples = df[df['isco_code'] == rare_class].index.tolist()
        if len(rare_samples) == 1:
            train_indices.extend(rare_samples)
        else:
            train_indices.append(rare_samples[0])
            test_indices.extend(rare_samples[1:])
    
    # 常见类别：正常分层抽样
    if common_classes:
        common_df = df[df['isco_code'].isin(common_classes)]
        train_common, test_common = train_test_split(
            common_df, test_size=test_size, random_state=seed,
            stratify=common_df['isco_code']
        )
        train_indices.extend(train_common.index.tolist())
        test_indices.extend(test_common.index.tolist())
    
    # 创建训练和测试集
    train_df = df.loc[train_indices]
    test_df = df.loc[test_indices]
    
    print(f"✅ 数据划分完成:")
    print(f"   训练集: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   测试集: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # 验证所有类别都在训练集中
    train_classes = set(train_df['isco_code'])
    all_classes = set(df['isco_code'])
    if train_classes == all_classes:
        print(f"✅ 训练集包含所有 {len(all_classes)} 个ISCO编码")
    else:
        missing = all_classes - train_classes
        print(f"⚠️  训练集缺失编码: {missing}")
    
    # 创建层次结构
    hierarchy = create_isco_hierarchy_from_codes(df['isco_code'].unique())
    
    # 保存数据文件
    output_dir = os.path.dirname(csv_path)
    train_texts_path = os.path.join(output_dir, "train_texts.txt")
    train_labels_path = os.path.join(output_dir, "train_labels.txt")
    test_texts_path = os.path.join(output_dir, "test_texts.txt")
    test_labels_path = os.path.join(output_dir, "test_labels.txt")
    
    save_as_text(train_texts_path, train_df['combined_text'].tolist())
    save_as_text(train_labels_path, train_df['isco_code'].tolist())
    save_as_text(test_texts_path, test_df['combined_text'].tolist())
    save_as_text(test_labels_path, test_df['isco_code'].tolist())
    
    if verbose:
        print(f"💾 数据文件已保存:")
        print(f"   {train_texts_path}")
        print(f"   {train_labels_path}")
        print(f"   {test_texts_path}")
        print(f"   {test_labels_path}")
    
    return (train_df['combined_text'].tolist(), 
            train_df['isco_code'].tolist(),
            hierarchy,
            test_df['combined_text'].tolist(),
            test_df['isco_code'].tolist())


def create_isco_hierarchy_from_codes(isco_codes):
    """从ISCO编码创建层次结构"""
    hierarchy = {}
    
    for code in isco_codes:
        code_str = str(code).zfill(4)
        
        # 创建各级编码
        for level in [1, 2, 3, 4]:
            level_code = code_str[:level]
            if level_code not in hierarchy:
                hierarchy[level_code] = create_hierarchy_node(
                    level_code, 
                    f"ISCO-{level}位-{level_code}"
                )
    
    return hierarchy


if __name__ == "__main__":
    main()