#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
职业分类工具函数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter


def create_hierarchy_node(class_str: str, name: str) -> Dict:
    """
    创建层次结构节点
    
    Args:
        class_str: ISCO编码字符串
        name: 节点名称
        
    Returns:
        层次结构节点字典
    """
    parents = get_parents(class_str)
    return {
        "parents": parents,
        "label": class_str,
        "name": name,
        "level": len(class_str)  # 根据编码长度确定层级
    }


def get_parents(class_str: str) -> List[str]:
    """
    获取ISCO编码的父级编码
    
    Args:
        class_str: ISCO编码字符串
        
    Returns:
        父级编码列表
    """
    parents = []
    for digit in [1, 2, 3]:  # 对于4位编码，父级为1、2、3位
        if len(class_str) > digit:
            parents.append(class_str[:digit])
    return parents


def create_hierarchy_from_isco_codes(isco_codes: List[str]) -> Dict:
    """
    从ISCO编码列表创建层次结构
    
    Args:
        isco_codes: ISCO编码列表
        
    Returns:
        层次结构字典
    """
    hierarchy = {}
    
    # 确保所有编码都是字符串格式
    isco_codes = [str(code).zfill(4) for code in isco_codes if pd.notna(code)]
    unique_codes = sorted(set(isco_codes))
    
    # 为每个唯一编码创建层次结构
    all_codes_to_create = set()
    
    for code in unique_codes:
        # 添加当前编码
        all_codes_to_create.add(code)
        
        # 添加所有父级编码
        for level in [1, 2, 3]:
            if len(code) > level:
                parent_code = code[:level]
                all_codes_to_create.add(parent_code)
    
    # 创建层次结构节点
    for code in sorted(all_codes_to_create):
        if code not in hierarchy:
            hierarchy[code] = create_hierarchy_node(
                code, 
                f"ISCO-{len(code)}位-{code}"
            )
    
    return hierarchy


def analyze_isco_distribution(df: pd.DataFrame, 
                            isco_field: str = 'ISCO_4_Digit_Code_Gemini') -> Dict:
    """
    分析ISCO编码分布
    
    Args:
        df: 数据框
        isco_field: ISCO编码字段名
        
    Returns:
        分析结果字典
    """
    print(f"📊 ISCO编码分布分析:")
    
    # 基本统计
    total_samples = len(df)
    unique_codes = df[isco_field].nunique()
    
    print(f"   总样本数: {total_samples:,}")
    print(f"   唯一ISCO编码: {unique_codes}")
    
    # 各级别统计
    df_copy = df.copy()
    df_copy['isco_str'] = df_copy[isco_field].astype(str).str.zfill(4)
    
    level_stats = {}
    for level in [1, 2, 3, 4]:
        level_codes = df_copy['isco_str'].str[:level].nunique()
        level_stats[level] = level_codes
        print(f"   {level}位编码数量: {level_codes}")
    
    # 频率分布
    code_counts = df[isco_field].value_counts()
    print(f"\n📈 频率分布:")
    print(f"   最频繁的5个编码:")
    for code, count in code_counts.head().items():
        print(f"     {code}: {count} 次 ({count/total_samples*100:.2f}%)")
    
    # 稀有编码统计
    rare_threshold = 5
    rare_codes = code_counts[code_counts <= rare_threshold]
    print(f"\n⚠️  稀有编码 (≤{rare_threshold}个样本):")
    print(f"   数量: {len(rare_codes)}")
    print(f"   占比: {len(rare_codes)/unique_codes*100:.2f}%")
    
    return {
        'total_samples': total_samples,
        'unique_codes': unique_codes,
        'level_stats': level_stats,
        'code_counts': code_counts,
        'rare_codes': rare_codes
    }


def prepare_chinese_job_text(df: pd.DataFrame, 
                            text_columns: List[str] = ['岗位', '岗位描述', '岗位职能']) -> pd.DataFrame:
    """
    准备中文职业文本数据
    
    Args:
        df: 数据框
        text_columns: 要组合的文本列
        
    Returns:
        处理后的数据框
    """
    print(f"📝 准备中文职业文本数据...")
    
    def combine_text_features(row):
        parts = []
        for col in text_columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                parts.append(str(row[col]).strip())
        return ' '.join(parts)
    
    # 组合文本
    df['combined_text'] = df.apply(combine_text_features, axis=1)
    
    # 文本质量检查
    text_lengths = df['combined_text'].str.len()
    print(f"   平均文本长度: {text_lengths.mean():.0f} 字符")
    print(f"   最短文本: {text_lengths.min()} 字符")
    print(f"   最长文本: {text_lengths.max()} 字符")
    
    # 检查空文本
    empty_texts = df['combined_text'].str.strip().eq('').sum()
    if empty_texts > 0:
        print(f"⚠️  发现 {empty_texts} 个空文本")
        # 移除空文本
        df = df[df['combined_text'].str.strip() != ''].copy()
        print(f"   移除后剩余: {len(df)} 条记录")
    
    return df


def validate_data_quality(df: pd.DataFrame, 
                         text_col: str = 'combined_text', 
                         label_col: str = 'ISCO_4_Digit_Code_Gemini') -> List[str]:
    """
    验证数据质量
    
    Args:
        df: 数据框
        text_col: 文本列名
        label_col: 标签列名
        
    Returns:
        问题列表
    """
    print(f"🔍 数据质量验证...")
    
    issues = []
    
    # 检查缺失值
    if df[text_col].isnull().sum() > 0:
        issues.append(f"文本列有 {df[text_col].isnull().sum()} 个缺失值")
    
    if df[label_col].isnull().sum() > 0:
        issues.append(f"标签列有 {df[label_col].isnull().sum()} 个缺失值")
    
    # 检查空文本
    empty_texts = df[text_col].str.strip().eq('').sum()
    if empty_texts > 0:
        issues.append(f"有 {empty_texts} 个空文本")
    
    # 检查文本长度
    very_short = (df[text_col].str.len() < 10).sum()
    if very_short > 0:
        issues.append(f"有 {very_short} 个过短文本 (<10字符)")
    
    very_long = (df[text_col].str.len() > 1000).sum()
    if very_long > 0:
        issues.append(f"有 {very_long} 个过长文本 (>1000字符)")
    
    # 检查标签格式
    invalid_labels = df[label_col].astype(str).str.len().ne(4).sum()
    if invalid_labels > 0:
        issues.append(f"有 {invalid_labels} 个非4位ISCO编码")
    
    if issues:
        print(f"⚠️  发现数据质量问题:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print(f"✅ 数据质量良好")
    
    return issues


def split_data_by_hierarchy(df: pd.DataFrame, 
                           label_col: str = 'isco_code',
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    考虑层次结构的数据划分
    
    Args:
        df: 数据框
        label_col: 标签列名
        test_size: 测试集比例
        random_state: 随机种子
        
    Returns:
        训练集和测试集
    """
    from sklearn.model_selection import train_test_split
    
    # 统计类别分布
    label_counts = Counter(df[label_col])
    
    # 分离单样本和多样本类别
    single_sample_classes = [label for label, count in label_counts.items() if count == 1]
    multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
    
    print(f"📊 数据划分:")
    print(f"   单样本类别: {len(single_sample_classes)}")
    print(f"   多样本类别: {len(multi_sample_classes)}")
    
    # 单样本类别全部放入训练集
    train_dfs = []
    test_dfs = []
    
    if single_sample_classes:
        single_df = df[df[label_col].isin(single_sample_classes)]
        train_dfs.append(single_df)
    
    # 多样本类别进行分层划分
    if multi_sample_classes:
        multi_df = df[df[label_col].isin(multi_sample_classes)]
        
        try:
            train_multi, test_multi = train_test_split(
                multi_df, 
                test_size=test_size, 
                random_state=random_state,
                stratify=multi_df[label_col]
            )
        except ValueError:
            # 分层失败，使用随机划分
            train_multi, test_multi = train_test_split(
                multi_df, 
                test_size=test_size, 
                random_state=random_state
            )
        
        train_dfs.append(train_multi)
        test_dfs.append(test_multi)
    
    # 合并结果
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()
    
    print(f"   训练集: {len(train_df)} 样本")
    print(f"   测试集: {len(test_df)} 样本")
    
    return train_df, test_df


def calculate_hierarchical_metrics(y_true: List[str], 
                                  y_pred: List[str]) -> Dict:
    """
    计算层次化评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        各级别的准确率
    """
    from sklearn.metrics import accuracy_score
    
    metrics = {}
    
    # 计算各级别准确率
    for level in [1, 2, 3, 4]:
        true_level = [label[:level] for label in y_true]
        pred_level = [label[:level] for label in y_pred]
        
        accuracy = accuracy_score(true_level, pred_level)
        metrics[f'level_{level}_accuracy'] = accuracy
    
    # 计算错误分布
    error_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for true, pred in zip(y_true, y_pred):
        if true != pred:
            # 找出最高不同级别
            for level in [1, 2, 3, 4]:
                if true[:level] != pred[:level]:
                    error_counts[level] += 1
                    break
    
    metrics['error_distribution'] = error_counts
    
    return metrics


# 测试代码
if __name__ == "__main__":
    print("🔧 职业分类工具函数")
    print("=" * 50)
    
    # 测试层次结构创建
    print("\n测试层次结构创建:")
    test_codes = ['1121', '1122', '2121', '2122', '3111']
    hierarchy = create_hierarchy_from_isco_codes(test_codes)
    
    print(f"创建了 {len(hierarchy)} 个节点")
    for code, node in sorted(hierarchy.items())[:5]:
        print(f"  {code}: level={node['level']}, parents={node['parents']}")
    
    # 测试层次化指标计算
    print("\n测试层次化指标计算:")
    y_true = ['1121', '1122', '2121', '2122']
    y_pred = ['1122', '1121', '2121', '1122']  # 第4个是1级错误
    
    metrics = calculate_hierarchical_metrics(y_true, y_pred)
    print("各级别准确率:")
    for level in [1, 2, 3, 4]:
        acc = metrics[f'level_{level}_accuracy']
        print(f"  {level}级: {acc:.2%}")
    
    print("\n错误分布:")
    for level, count in metrics['error_distribution'].items():
        print(f"  {level}级错误: {count} 个")