#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文职业分类数据增强与预处理模块
专为ISCO编号识别任务设计的增强版预处理器
"""

import re
import jieba
import jieba.posseg as pseg
import pandas as pd
import numpy as np
import random
import ast
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

class ChineseJobTextProcessor:
    """中文职业文本预处理器"""
    
    def __init__(self, 
                 stopwords_path: Optional[str] = None,
                 job_synonyms_path: Optional[str] = None,
                 industry_terms_path: Optional[str] = None):
        
        # 加载停用词
        self.stopwords = self._load_stopwords(stopwords_path)
        
        # 加载职业同义词字典
        self.job_synonyms = self._load_job_synonyms(job_synonyms_path)
        
        # 加载行业术语标准化字典
        self.industry_terms = self._load_industry_terms(industry_terms_path)
        
        # 预编译正则表达式
        self._compile_patterns()
        
        # 初始化jieba
        self._init_jieba()
        
        print("✅ 中文职业文本处理器初始化完成")

    def _load_stopwords(self, path: Optional[str]) -> set:
        """加载停用词"""
        if path and Path(path).exists():
            with open(path, 'r', encoding='utf-8') as f:
                return set(line.strip() for line in f if line.strip())
        
        # 默认停用词（针对职业描述优化）
        return {
            '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '上', '也', '很', '到', '要', '会', '着', '看', '好', '自己',
            '这', '那', '里', '为', '他', '她', '它', '但', '而', '或', '与', '及', '以', '可以', '能够', '应该', '需要', '通过',
            '由于', '因为', '所以', '如果', '虽然', '然而', '因此', '于是', '并且', '而且', '但是', '相关', '等等', '包括',
            '具有', '具备', '拥有', '熟悉', '了解', '掌握', '优先', '优秀', '良好', '较强', '能力', '工作', '经验', '要求',
            '岗位', '职位', '任职', '从事', '负责', '完成', '进行', '组织', '协调', '管理', '建立', '制定', '实施', '执行'
        }

    def _load_job_synonyms(self, path: Optional[str]) -> Dict[str, List[str]]:
        """加载职业同义词字典"""
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 默认职业同义词字典
        return {
            'CEO': ['首席执行官', '总裁', '总经理', '执行总裁', '董事长'],
            '人事': ['人力资源', '人事管理', 'HR', '人员管理', '人事行政'],
            '采购': ['采购管理', '供应链', '物料采购', '商品采购', '战略采购'],
            '销售': ['营销', '市场销售', '业务', '商务', '客户经理'],
            '财务': ['会计', '财务管理', '资金管理', '成本控制', '审计'],
            '技术': ['研发', '开发', '工程师', '技术开发', '软件开发'],
            '运营': ['运营管理', '业务运营', '产品运营', '数据运营'],
            '行政': ['行政管理', '办公室管理', '后勤', '综合管理'],
            '培训': ['培训管理', '人才发展', '学习发展', '教育培训'],
            '质量': ['质量管理', '品质控制', 'QA', 'QC', '质量保证'],
            '项目': ['项目管理', '项目经理', '项目主管', '项目协调'],
            '客服': ['客户服务', '售后服务', '客户支持', '服务管理']
        }

    def _load_industry_terms(self, path: Optional[str]) -> Dict[str, str]:
        """加载行业术语标准化字典"""
        if path and Path(path).exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 默认行业术语标准化
        return {
            'IT': '信息技术', 'AI': '人工智能', 'ERP': '企业资源规划',
            'CRM': '客户关系管理', 'SCM': '供应链管理', 'OA': '办公自动化',
            'KPI': '关键绩效指标', 'ROI': '投资回报率', 'SOP': '标准作业程序',
            'B2B': '企业对企业', 'B2C': '企业对消费者', 'O2O': '线上到线下',
            'UI': '用户界面', 'UE': '用户体验', 'UX': '用户体验',
            'API': '应用程序接口', 'SDK': '软件开发工具包',
            'SQL': '结构化查询语言', 'HTML': '超文本标记语言',
            'CEO': '首席执行官', 'CFO': '首席财务官', 'CTO': '首席技术官',
            'COO': '首席运营官', 'CHO': '首席人力资源官', 'CMO': '首席营销官'
        }

    def _compile_patterns(self):
        """预编译正则表达式模式"""
        # 清理模式
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'1[3-9]\d{9}|0\d{2,3}-?\d{7,8}')
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.number_pattern = re.compile(r'\d+')
        
        # 职业层级关键词
        self.level_patterns = {
            '高级': re.compile(r'(高级|资深|首席|总|副总|主任|经理|总监|董事)'),
            '中级': re.compile(r'(主管|副主管|组长|leader|负责人)'),
            '初级': re.compile(r'(专员|助理|文员|实习|见习|初级)')
        }
        
        # 技能关键词模式
        self.skill_pattern = re.compile(r'(熟练|熟悉|掌握|精通|了解|会|能够|具备).{0,20}(软件|系统|工具|技能|技术|语言|平台)')

    def _init_jieba(self):
        """初始化jieba分词器并添加职业相关词汇"""
        # 添加职业相关词汇到jieba词典
        job_terms = [
            '首席执行官', '首席财务官', '首席技术官', '人力资源',
            '市场营销', '客户服务', '供应链管理', '项目管理',
            '数据分析', '软件开发', '产品经理', '运营管理'
        ]
        
        for term in job_terms:
            jieba.add_word(term)

    def clean_text(self, text: str) -> str:
        """文本清理"""
        if pd.isna(text) or not text:
            return ""
        
        text = str(text)
        
        # 移除邮箱、电话、网址
        text = self.email_pattern.sub('', text)
        text = self.phone_pattern.sub('', text)
        text = self.url_pattern.sub('', text)
        
        # 标准化空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除特殊字符但保留中文、英文、数字和基本标点
        text = re.sub(r'[^\u4e00-\u9fff\u0041-\u005a\u0061-\u007a0-9\s，。、；：""''（）【】]', '', text)
        
        return text

    def extract_structure_info(self, text: str) -> Dict[str, str]:
        """提取职位描述的结构化信息"""
        structure = {
            'responsibilities': '',  # 岗位职责
            'requirements': '',      # 任职要求
            'skills': '',           # 技能要求
            'experience': '',       # 经验要求
            'education': ''         # 学历要求
        }
        
        # 职责关键词
        resp_keywords = ['职责', '工作内容', '主要职责', '具体工作', '工作职责']
        req_keywords = ['要求', '任职要求', '岗位要求', '招聘要求', '应聘要求']
        skill_keywords = ['技能', '能力', '专业技能', '技术要求']
        exp_keywords = ['经验', '工作经验', '从业经验', '相关经验']
        edu_keywords = ['学历', '教育', '专业', '毕业']
        
        # 分段提取
        sections = re.split(r'[1-9]、|[1-9]\.|[一二三四五六七八九十]、', text)
        
        for section in sections:
            section_lower = section.lower()
            
            # 判断段落类型并提取相关内容
            if any(kw in section for kw in resp_keywords):
                structure['responsibilities'] += section + ' '
            elif any(kw in section for kw in req_keywords):
                structure['requirements'] += section + ' '
            elif any(kw in section for kw in skill_keywords):
                structure['skills'] += section + ' '
            elif any(kw in section for kw in exp_keywords):
                structure['experience'] += section + ' '
            elif any(kw in section for kw in edu_keywords):
                structure['education'] += section + ' '
        
        # 清理和截断
        for key in structure:
            structure[key] = structure[key].strip()[:200]  # 限制长度
        
        return structure

    def normalize_industry_terms(self, text: str) -> str:
        """标准化行业术语"""
        for abbr, full in self.industry_terms.items():
            # 替换独立的缩写词
            text = re.sub(r'\b' + re.escape(abbr) + r'\b', full, text, flags=re.IGNORECASE)
        
        return text

    def segment_text(self, text: str, extract_pos: bool = True) -> List[str]:
        """中文分词"""
        if extract_pos:
            # 使用词性标注，只保留有意义的词
            words = []
            for word, flag in pseg.cut(text):
                # 保留名词、动词、形容词、专有名词等
                if (flag.startswith(('n', 'v', 'a', 'nr', 'nt', 'nz', 'eng')) 
                    and len(word.strip()) > 1 
                    and word not in self.stopwords):
                    words.append(word)
            return words
        else:
            # 普通分词
            words = jieba.lcut(text)
            return [w for w in words if len(w.strip()) > 1 and w not in self.stopwords]

    def extract_features(self, text: str) -> Dict[str, any]:
        """提取文本特征"""
        features = {}
        
        # 基础统计特征
        features['text_length'] = len(text)
        features['word_count'] = len(self.segment_text(text, extract_pos=False))
        
        # 职业层级特征
        for level, pattern in self.level_patterns.items():
            features[f'has_{level}_level'] = bool(pattern.search(text))
        
        # 技能要求特征
        skill_matches = self.skill_pattern.findall(text)
        features['skill_count'] = len(skill_matches)
        features['has_skill_requirements'] = len(skill_matches) > 0
        
        # 数字特征（年限、薪资等）
        numbers = self.number_pattern.findall(text)
        features['number_count'] = len(numbers)
        
        # 关键词特征
        key_terms = ['管理', '销售', '技术', '财务', '人事', '市场', '客服', '采购', '运营', '行政']
        for term in key_terms:
            features[f'has_{term}'] = term in text
        
        return features


class ChineseJobDataAugmenter:
    """中文职业数据增强器"""
    
    def __init__(self, processor: ChineseJobTextProcessor):
        self.processor = processor
        
        # 增强策略配置
        self.augmentation_strategies = {
            'synonym_replacement': 0.3,    # 同义词替换概率
            'structure_reorder': 0.2,      # 结构重排概率  
            'keyword_enhancement': 0.25,   # 关键词增强概率
            'paraphrase': 0.15,           # 语言重组概率
            'noise_injection': 0.1         # 噪声注入概率
        }
        
        print("✅ 中文职业数据增强器初始化完成")

    def synonym_replacement(self, text: str, replacement_rate: float = 0.15) -> str:
        """同义词替换增强"""
        words = self.processor.segment_text(text, extract_pos=False)
        
        # 计算需要替换的词数
        num_replace = max(1, int(len(words) * replacement_rate))
        
        # 随机选择要替换的位置
        replace_indices = random.sample(range(len(words)), min(num_replace, len(words)))
        
        for idx in replace_indices:
            word = words[idx]
            
            # 查找同义词
            for key, synonyms in self.processor.job_synonyms.items():
                if word in synonyms:
                    # 随机选择一个同义词替换
                    replacement = random.choice([s for s in synonyms if s != word])
                    words[idx] = replacement
                    break
        
        return ''.join(words)

    def structure_reorder(self, text: str) -> str:
        """结构重排增强"""
        # 提取结构化信息
        structure = self.processor.extract_structure_info(text)
        
        # 重新组织顺序
        sections = []
        for key, content in structure.items():
            if content.strip():
                sections.append(content)
        
        if len(sections) > 1:
            # 随机打乱部分段落顺序
            random.shuffle(sections)
            return ' '.join(sections)
        
        return text

    def keyword_enhancement(self, text: str) -> str:
        """关键词增强"""
        # 提取关键词
        words = self.processor.segment_text(text, extract_pos=True)
        
        # 识别职业相关关键词
        job_keywords = []
        for word in words:
            for category, synonyms in self.processor.job_synonyms.items():
                if word in synonyms:
                    job_keywords.append(word)
                    break
        
        # 随机重复一些关键词（模拟重要性强调）
        if job_keywords:
            enhanced_keyword = random.choice(job_keywords)
            # 在文本末尾添加关键词强调
            text += f" 重点关注{enhanced_keyword}相关工作"
        
        return text

    def paraphrase_text(self, text: str) -> str:
        """语言重组增强"""
        # 简单的语言重组策略
        sentences = re.split(r'[。；！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) > 1:
            # 随机调整句子顺序
            random.shuffle(sentences)
            return '。'.join(sentences) + '。'
        
        return text

    def inject_noise(self, text: str, noise_rate: float = 0.05) -> str:
        """噪声注入增强"""
        chars = list(text)
        num_noise = max(1, int(len(chars) * noise_rate))
        
        # 随机插入或删除少量字符
        for _ in range(num_noise):
            if random.random() < 0.5 and len(chars) > 10:
                # 删除字符
                idx = random.randint(0, len(chars) - 1)
                if chars[idx] not in '。，；：！？':  # 保留重要标点
                    chars.pop(idx)
            else:
                # 插入空格（模拟OCR错误）
                idx = random.randint(0, len(chars))
                chars.insert(idx, ' ')
        
        return ''.join(chars)

    def augment_single_text(self, text: str, num_augmented: int = 3) -> List[str]:
        """对单个文本进行增强"""
        augmented_texts = [text]  # 包含原文
        
        strategies = list(self.augmentation_strategies.keys())
        
        for _ in range(num_augmented):
            current_text = text
            
            # 随机选择1-2个增强策略组合使用
            selected_strategies = random.sample(strategies, random.randint(1, 2))
            
            for strategy in selected_strategies:
                if random.random() < self.augmentation_strategies[strategy]:
                    if strategy == 'synonym_replacement':
                        current_text = self.synonym_replacement(current_text)
                    elif strategy == 'structure_reorder':
                        current_text = self.structure_reorder(current_text)
                    elif strategy == 'keyword_enhancement':
                        current_text = self.keyword_enhancement(current_text)
                    elif strategy == 'paraphrase':
                        current_text = self.paraphrase_text(current_text)
                    elif strategy == 'noise_injection':
                        current_text = self.inject_noise(current_text)
            
            # 清理增强后的文本
            current_text = self.processor.clean_text(current_text)
            
            if current_text and current_text != text:  # 确保增强后文本有效且不同
                augmented_texts.append(current_text)
        
        return augmented_texts


class ImbalancedDataHandler:
    """不平衡数据处理器"""
    
    def __init__(self, min_samples_per_class: int = 5):
        self.min_samples_per_class = min_samples_per_class
        
    def analyze_class_distribution(self, labels: List[str]) -> Dict[str, int]:
        """分析类别分布"""
        distribution = Counter(labels)
        
        print(f"📊 类别分布分析:")
        print(f"   总类别数: {len(distribution)}")
        print(f"   总样本数: {sum(distribution.values())}")
        print(f"   平均每类样本数: {sum(distribution.values()) / len(distribution):.1f}")
        
        # 分类统计
        rare_classes = {k: v for k, v in distribution.items() if v < self.min_samples_per_class}
        medium_classes = {k: v for k, v in distribution.items() if self.min_samples_per_class <= v < 20}
        common_classes = {k: v for k, v in distribution.items() if v >= 20}
        
        print(f"   稀有类别 (<{self.min_samples_per_class}): {len(rare_classes)}")
        print(f"   中等类别 ({self.min_samples_per_class}-19): {len(medium_classes)}")
        print(f"   常见类别 (>=20): {len(common_classes)}")
        
        return {
            'all': distribution,
            'rare': rare_classes,
            'medium': medium_classes,
            'common': common_classes
        }
    
    def balance_dataset(self, 
                       texts: List[str], 
                       labels: List[str], 
                       augmenter: ChineseJobDataAugmenter,
                       target_samples: int = 10) -> Tuple[List[str], List[str]]:
        """平衡数据集"""
        
        distribution = self.analyze_class_distribution(labels)
        
        balanced_texts = []
        balanced_labels = []
        
        # 按类别分组
        class_data = defaultdict(list)
        for text, label in zip(texts, labels):
            class_data[label].append(text)
        
        print(f"\n🔄 开始数据平衡处理...")
        
        for label, class_texts in class_data.items():
            current_count = len(class_texts)
            
            # 添加原始数据
            balanced_texts.extend(class_texts)
            balanced_labels.extend([label] * current_count)
            
            # 如果需要增强
            if current_count < target_samples:
                needed = target_samples - current_count
                print(f"   {label}: 原有{current_count}个样本，需增强{needed}个")
                
                # 对稀有类别进行更强的增强
                augment_per_sample = max(1, needed // current_count + 1)
                
                augmented_count = 0
                for text in class_texts:
                    if augmented_count >= needed:
                        break
                    
                    # 生成增强样本
                    augmented = augmenter.augment_single_text(text, augment_per_sample)
                    
                    # 跳过原文，只要增强样本
                    for aug_text in augmented[1:]:
                        if augmented_count < needed:
                            balanced_texts.append(aug_text)
                            balanced_labels.append(label)
                            augmented_count += 1
        
        print(f"✅ 数据平衡完成:")
        print(f"   原始样本: {len(texts)}")
        print(f"   平衡后样本: {len(balanced_texts)}")
        
        return balanced_texts, balanced_labels


class EnhancedJobDataProcessor:
    """增强版职业数据处理器 - 整合所有功能"""
    
    def __init__(self, 
                 stopwords_path: Optional[str] = None,
                 job_synonyms_path: Optional[str] = None,
                 industry_terms_path: Optional[str] = None):
        
        # 初始化各组件
        self.text_processor = ChineseJobTextProcessor(
            stopwords_path, job_synonyms_path, industry_terms_path
        )
        self.augmenter = ChineseJobDataAugmenter(self.text_processor)
        self.balance_handler = ImbalancedDataHandler()
        
        print("🚀 增强版职业数据处理器初始化完成")

    def safe_train_test_split(self, df, test_size=0.2, random_state=42):
        """安全的训练测试集划分，处理单样本类别"""
        from sklearn.model_selection import train_test_split
        
        # 检查类别分布
        label_counts = Counter(df['isco_code'])
        single_sample_classes = [label for label, count in label_counts.items() if count == 1]
        multi_sample_classes = [label for label, count in label_counts.items() if count > 1]
        
        print(f"   检测到 {len(single_sample_classes)} 个单样本类别")
        print(f"   检测到 {len(multi_sample_classes)} 个多样本类别")
        
        train_dfs = []
        test_dfs = []
        
        # 处理单样本类别 - 全部放入训练集
        if single_sample_classes:
            single_df = df[df['isco_code'].isin(single_sample_classes)]
            train_dfs.append(single_df)
            print(f"   单样本类别 {len(single_sample_classes)} 个已加入训练集")
        
        # 处理多样本类别 - 正常分层划分
        if multi_sample_classes:
            multi_df = df[df['isco_code'].isin(multi_sample_classes)]
            
            try:
                train_multi, test_multi = train_test_split(
                    multi_df, test_size=test_size, random_state=random_state,
                    stratify=multi_df['isco_code']
                )
                train_dfs.append(train_multi)
                test_dfs.append(test_multi)
                print(f"   多样本类别成功分层划分")
            except ValueError as e:
                print(f"   分层划分失败: {e}")
                print(f"   使用随机划分")
                train_multi, test_multi = train_test_split(
                    multi_df, test_size=test_size, random_state=random_state
                )
                train_dfs.append(train_multi)
                test_dfs.append(test_multi)
        
        # 合并结果
        if train_dfs:
            train_df = pd.concat(train_dfs, ignore_index=True)
        else:
            train_df = pd.DataFrame()
        
        if test_dfs:
            test_df = pd.concat(test_dfs, ignore_index=True)
        else:
            # 如果没有测试集，从训练集中分出一部分
            if len(train_df) >= 10:
                split_point = max(1, len(train_df) // 5)
                test_df = train_df.tail(split_point).copy()
                train_df = train_df.head(len(train_df) - split_point).copy()
                print(f"   从训练集中分出 {len(test_df)} 个样本作为测试集")
            else:
                test_df = pd.DataFrame()
        
        return train_df, test_df

    def process_csv_data(self, 
                        csv_path: str,
                        text_columns: List[str] = ['岗位', '岗位描述', '岗位职能'],
                        label_column: str = 'ISCO_4_Digit_Code_Gemini',
                        enable_augmentation: bool = True,
                        balance_data: bool = True,
                        target_samples_per_class: int = 10) -> Tuple[List[str], List[str], Dict]:
        """处理CSV数据"""
        
        print(f"📊 开始处理CSV数据: {csv_path}")
        
        # 加载数据
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='gbk')
        
        print(f"   原始数据: {len(df)} 行")
        
        # 验证必需列
        missing_cols = [col for col in text_columns + [label_column] if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必需的列: {missing_cols}")
        
        # 处理岗位职能列（可能是字符串形式的列表）
        if '岗位职能' in text_columns:
            df['岗位职能'] = df['岗位职能'].apply(self._parse_job_functions)
        
        # 组合文本特征
        def combine_text_features(row):
            parts = []
            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    content = str(row[col])
                    # 清理和标准化
                    content = self.text_processor.clean_text(content)
                    content = self.text_processor.normalize_industry_terms(content)
                    if content:
                        parts.append(content)
            return ' '.join(parts)
        
        # 处理文本
        print("🔧 文本预处理中...")
        df['processed_text'] = df.apply(combine_text_features, axis=1)
        
        # 移除空文本
        initial_count = len(df)
        df = df[df['processed_text'].str.strip() != ''].copy()
        if len(df) < initial_count:
            print(f"   移除空文本: {initial_count - len(df)} 条")
        
        # 处理标签
        df['isco_code'] = df[label_column].astype(str).str.zfill(4)
        
        # 提取文本和标签
        texts = df['processed_text'].tolist()
        labels = df['isco_code'].tolist()
        
        # 数据质量分析
        stats = self._analyze_data_quality(texts, labels)
        
        # 数据增强和平衡
        if enable_augmentation and balance_data:
            print("🎯 开始数据增强和平衡...")
            texts, labels = self.balance_handler.balance_dataset(
                texts, labels, self.augmenter, target_samples_per_class
            )
            
        elif enable_augmentation:
            print("🎯 开始数据增强...")
            augmented_texts = []
            augmented_labels = []
            
            for text, label in zip(texts, labels):
                aug_texts = self.augmenter.augment_single_text(text, num_augmented=2)
                augmented_texts.extend(aug_texts)
                augmented_labels.extend([label] * len(aug_texts))
            
            texts, labels = augmented_texts, augmented_labels
        
        # 最终统计
        final_stats = self._analyze_data_quality(texts, labels)
        
        return texts, labels, {
            'original_stats': stats,
            'final_stats': final_stats,
            'processing_info': {
                'augmentation_enabled': enable_augmentation,
                'balance_enabled': balance_data,
                'target_samples_per_class': target_samples_per_class
            }
        }

    def _parse_job_functions(self, job_func_str):
        """解析岗位职能字符串"""
        if pd.isna(job_func_str):
            return ""
        
        job_func_str = str(job_func_str)
        
        try:
            # 尝试解析为Python列表
            if job_func_str.startswith('[') and job_func_str.endswith(']'):
                job_funcs = ast.literal_eval(job_func_str)
                return ' '.join(job_funcs) if isinstance(job_funcs, list) else str(job_func_str)
        except:
            pass
        
        return job_func_str

    def _analyze_data_quality(self, texts: List[str], labels: List[str]) -> Dict:
        """分析数据质量"""
        text_lengths = [len(text) for text in texts]
        word_counts = [len(self.text_processor.segment_text(text, extract_pos=False)) for text in texts]
        
        return {
            'total_samples': len(texts),
            'unique_labels': len(set(labels)),
            'avg_text_length': np.mean(text_lengths),
            'avg_word_count': np.mean(word_counts),
            'min_text_length': min(text_lengths) if text_lengths else 0,
            'max_text_length': max(text_lengths) if text_lengths else 0,
            'label_distribution': Counter(labels)
        }

    def save_processed_data(self, 
                           texts: List[str], 
                           labels: List[str], 
                           output_dir: str):
        """保存处理后的数据"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存文本和标签
        with open(output_path / 'processed_texts.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        with open(output_path / 'processed_labels.txt', 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(label + '\n')
        
        print(f"✅ 处理后数据已保存到: {output_dir}")


# 使用示例和测试
if __name__ == "__main__":
    
    # 创建处理器
    processor = EnhancedJobDataProcessor()
    
    # 示例：处理你的数据
    csv_path = "newjob1_sortall.csv"  # 你的数据文件路径
    
    try:
        # 处理数据
        texts, labels, stats = processor.process_csv_data(
            csv_path=csv_path,
            enable_augmentation=True,
            balance_data=True,
            target_samples_per_class=8
        )
        
        # 显示处理结果
        print(f"\n📈 处理结果统计:")
        print(f"原始数据: {stats['original_stats']['total_samples']} 样本")
        print(f"处理后数据: {stats['final_stats']['total_samples']} 样本")
        print(f"类别数量: {stats['final_stats']['unique_labels']}")
        print(f"平均文本长度: {stats['final_stats']['avg_text_length']:.1f} 字符")
        print(f"平均词数: {stats['final_stats']['avg_word_count']:.1f} 词")
        
        # 保存处理后的数据
        processor.save_processed_data(texts, labels, "processed_job_data")
        
        print("\n🎉 数据增强与预处理完成！")
        
    except FileNotFoundError:
        print(f"❌ 找不到数据文件: {csv_path}")
        print("请确认文件路径正确，或使用示例数据测试功能")
        
        # 创建示例数据进行测试
        sample_texts = [
            "负责公司人力资源管理工作，包括招聘、培训、绩效考核等",
            "协助CEO制定公司发展战略，管理高级团队",
            "负责采购管理，供应商关系维护，成本控制"
        ]
        sample_labels = ["1212", "1111", "1212"]
        
        print("\n🧪 使用示例数据测试增强功能...")
        
        # 测试单个文本增强
        augmented = processor.augmenter.augment_single_text(sample_texts[0], num_augmented=3)
        
        print(f"原文: {sample_texts[0]}")
        print("增强结果:")
        for i, aug_text in enumerate(augmented[1:], 1):
            print(f"  {i}. {aug_text}")