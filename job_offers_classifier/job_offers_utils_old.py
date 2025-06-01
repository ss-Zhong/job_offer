import pandas as pd
import numpy as np


def concat_to_str(x):
    """连接文本字段"""
    x = [str(x_i) for x_i in x if x_i == x_i]
    x = [x_i for x_i in x if len(x_i)]
    return ' '.join(x)


def fix_class_str(t):
    """修复类别字符串格式"""
    t = str(t)
    t = t.split('.')[0]
    t = '0' * (6 - len(t)) + t
    return t


def check_code(cl):
    """检查编码是否为6位"""
    return len(str(cl)) == 6


def class_to_digit(cl, digit=6):
    """提取指定位数的类别编码"""
    return str(cl)[:digit]


def get_parents(class_str):
    """获取ISCO编码的父级编码"""
    parents = []
    for digit in [1, 2, 3]:  # 对于4位编码，父级为1、2、3位
        if len(class_str) > digit:
            parents.append(class_str[:digit])
    return parents


def create_hierarchy_node(class_str, name):
    """创建层次结构节点"""
    parents = get_parents(class_str)
    return {
        "parents": parents, 
        "label": class_str, 
        "name": name, 
        "level": len(class_str)  # 根据编码长度确定层级
    }


def create_hierarchy(df, class_str_field='class', name_field='name'):
    """从DataFrame创建层次结构"""
    hierarchy = {}
    for i, row in df.iterrows():
        class_code = str(row[class_str_field])
        name = str(row[name_field]) if name_field in row and pd.notna(row[name_field]) else f"ISCO-{class_code}"
        hierarchy[class_code] = create_hierarchy_node(class_code, name)
    return hierarchy


def create_hierarchy_from_isco_codes(isco_codes):
    """从ISCO编码列表创建层次结构"""
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
            hierarchy[code] = create_hierarchy_node(code, f"ISCO-{len(code)}位-{code}")
    
    return hierarchy


def _return_new_dfs(new_dfs):
    """返回处理后的DataFrame"""
    if len(new_dfs) == 1:
        new_dfs = new_dfs[0]
    else:
        new_dfs = tuple(new_dfs)
    return new_dfs


def remove_classes(dfs, classes, class_field='class'):
    """移除指定的类别"""
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        for c in classes:
            index = new_df[class_field].astype(str).str.startswith(c)
            new_df = new_df[~index]
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def remap_classes(dfs, classes_map, class_field='class'):
    """重新映射类别"""
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        new_df[class_field] = new_df[class_field].apply(lambda x: classes_map.get(x, x))
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def filter_classes(dfs, classes, class_field='class'):
    """过滤指定的类别"""
    if not isinstance(dfs, (list, tuple)):
        dfs = [dfs]

    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        index = new_df[class_field].isin(classes)
        new_df = new_df[index]
        new_dfs.append(new_df)
    return _return_new_dfs(new_dfs)


def top_k_prediction(pred, top_k):
    """获取top-k预测结果"""
    top_k_labels = np.flip(np.argsort(pred, axis=1))[:, :top_k]
    top_k_prob = np.take_along_axis(pred, top_k_labels, axis=1)
    return top_k_labels, top_k_prob


def analyze_isco_distribution(df, isco_field='ISCO_4_Digit_Code_Gemini'):
    """分析ISCO编码分布"""
    print(f"📊 ISCO编码分布分析:")
    
    # 基本统计
    total_samples = len(df)
    unique_codes = df[isco_field].nunique()
    
    print(f"   总样本数: {total_samples:,}")
    print(f"   唯一ISCO编码: {unique_codes}")
    
    # 各级别统计
    df_copy = df.copy()
    df_copy['isco_str'] = df_copy[isco_field].astype(str).str.zfill(4)
    
    for level in [1, 2, 3, 4]:
        level_codes = df_copy['isco_str'].str[:level].nunique()
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
        'code_counts': code_counts,
        'rare_codes': rare_codes
    }


def prepare_chinese_job_text(df, text_columns=['岗位', '岗位描述', '岗位职能']):
    """准备中文职业文本数据"""
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


def validate_data_quality(df, text_col='combined_text', label_col='ISCO_4_Digit_Code_Gemini'):
    """验证数据质量"""
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