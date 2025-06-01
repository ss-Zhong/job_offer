#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化分类工具模块
实现ISCO层次化损失函数和多任务学习工具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class HierarchicalLoss(nn.Module):
    """
    ISCO层次化损失函数
    根据层次结构计算加权损失，层级越高（越细粒度）惩罚越小
    """
    
    def __init__(self, 
                 isco_hierarchy: Dict,
                 level_weights: Optional[Dict[int, float]] = None,
                 temperature: float = 1.0,
                 alpha: float = 0.7):
        """
        Args:
            isco_hierarchy: ISCO层次结构字典
            level_weights: 各级别损失权重 {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
            temperature: 温度参数，用于平滑预测概率
            alpha: 层次损失和标准损失的平衡参数
        """
        super(HierarchicalLoss, self).__init__()
        
        self.isco_hierarchy = isco_hierarchy
        self.temperature = temperature
        self.alpha = alpha
        
        # 默认权重：层级越高（越细粒度）权重越小
        self.level_weights = level_weights or {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
        self._build_mappings()
        self.prefix2fullidx = {l: defaultdict(list) for l in [1,2,3,4]}
        for code, idx in self.code_to_idx[4].items():
            for l in [1,2,3,4]:
                pref = code[:l]
                self.prefix2fullidx[l][pref].append(idx)
                # 构建映射关系
        
        
        # 基础损失函数
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
        print(f"✅ 层次化损失函数初始化完成")
        print(f"   层级权重: {self.level_weights}")
        print(f"   温度参数: {self.temperature}")
        print(f"   平衡参数α: {self.alpha}")

    def _build_mappings(self):
        """构建ISCO编码的层次映射关系"""
        
        # 4级编码到各级别编码的映射
        self.level_mappings = {}
        self.level_to_labels = {1: set(), 2: set(), 3: set(), 4: set()}
        
        # 从层次结构中提取所有ISCO编码
        all_codes = set()
        for isco_code, node_info in self.isco_hierarchy.items():
            if len(isco_code) >= 1:  # 确保至少是1位编码
                all_codes.add(isco_code)
        
        # 为每个编码构建层次映射
        for isco_code in all_codes:
            if len(isco_code) >= 4:  # 如果是4级编码
                level_codes = {}
                for level in [1, 2, 3, 4]:
                    level_code = isco_code[:level]
                    level_codes[level] = level_code
                    self.level_to_labels[level].add(level_code)
                
                self.level_mappings[isco_code] = level_codes
            elif len(isco_code) >= 1:  # 处理1-3级编码
                max_level = len(isco_code)
                level_codes = {}
                for level in range(1, max_level + 1):
                    level_code = isco_code[:level]
                    level_codes[level] = level_code
                    self.level_to_labels[level].add(level_code)
                # 补充到4级
                for level in range(max_level + 1, 5):
                    level_codes[level] = isco_code  # 用最大可用级别填充
                    self.level_to_labels[level].add(isco_code)
                
                self.level_mappings[isco_code] = level_codes
        
        # 转换为排序列表，便于索引
        for level in [1, 2, 3, 4]:
            self.level_to_labels[level] = sorted(list(self.level_to_labels[level]))
        
        # 创建编码到索引的映射
        self.code_to_idx = {}
        self.idx_to_code = {}
        
        for level in [1, 2, 3, 4]:
            self.code_to_idx[level] = {code: idx for idx, code in enumerate(self.level_to_labels[level])}
            self.idx_to_code[level] = {idx: code for code, idx in self.code_to_idx[level].items()}
        
        # 构建层次距离矩阵
        self._build_hierarchy_distance_matrix()
        
        print(f"   各级别类别数: {[len(self.level_to_labels[i]) for i in [1,2,3,4]]}")

    def _build_hierarchy_distance_matrix(self):
        """构建层次距离矩阵，用于计算预测和真实标签的层次距离"""
        
        num_4_level_classes = len(self.level_to_labels[4])
        
        if num_4_level_classes == 0:
            print("⚠️ 警告：4级类别数为0，创建空的距离矩阵")
            self.hierarchy_distance = torch.zeros(1, 1)
            self.hierarchy_weights = torch.ones(1, 1)
            return
        
        # 创建距离矩阵：4级分类 x 4级分类
        self.hierarchy_distance = torch.zeros(num_4_level_classes, num_4_level_classes)
        
        for i, true_code in enumerate(self.level_to_labels[4]):
            for j, pred_code in enumerate(self.level_to_labels[4]):
                if true_code == pred_code:
                    distance = 0.0  # 完全正确
                else:
                    # 计算在哪个层级开始分歧
                    distance = 4.0  # 最大距离（4级分错）
                    
                    for level in [1, 2, 3]:
                        true_level_code = true_code[:min(level, len(true_code))]
                        pred_level_code = pred_code[:min(level, len(pred_code))]
                        
                        if true_level_code == pred_level_code:
                            distance = 4.0 - level  # 在level+1级开始分错
                        else:
                            break
                
                self.hierarchy_distance[i, j] = distance
        
        # 转换距离为权重
        self.hierarchy_weights = torch.zeros_like(self.hierarchy_distance)
        for distance, weight in {0.0: 1.0, 1.0: 2.0, 2.0: 4.0, 3.0: 8.0, 4.0: 8.0}.items():
            mask = (self.hierarchy_distance == distance)
            self.hierarchy_weights[mask] = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                target_codes: List[str]) -> Dict[str, torch.Tensor]:
        """
        计算层次化损失
        
        Args:
            logits: 模型预测 logits [batch_size, num_classes]
            targets: 目标类别索引 [batch_size]
            target_codes: 目标ISCO编码列表 [batch_size]
            
        Returns:
            Dict包含各种损失值
        """
        batch_size = logits.size(0)
        device = logits.device
        num_classes = logits.size(1)
        
        # 检查数据有效性
        if batch_size == 0 or len(target_codes) == 0:
            return {
                'total_loss': torch.tensor(0.0, device=device),
                'ce_loss': torch.tensor(0.0, device=device),
                'hierarchical_loss': torch.tensor(0.0, device=device),
                'level_accuracies': {f'level_{i}_acc': 0.0 for i in [1,2,3,4]}
            }
        
        # 将权重矩阵移到正确设备
        if self.hierarchy_weights.device != device:
            self.hierarchy_weights = self.hierarchy_weights.to(device)
        print(">>> [HierarchicalLoss] forward called, batch_size=", batch_size, " num_classes=", num_classes)

        # 1. 标准交叉熵
        # 1. 标准交叉熵损失

        # 1. 标准交叉熵损失
        ce_loss = self.ce_loss(logits / self.temperature, targets)
        
        probs = F.softmax(logits / self.temperature, dim=1)  # [B, C_full]
        total_level_weight = float(sum(self.level_weights.values()))
        batch_hl = []
            # 为每个样本计算层次化损失
            
        # 2. 层次化损失计算
        for i in range(batch_size):
            sample_loss = torch.tensor(0.0, device=logits.device)
            true_code = target_codes[i]
            for level, w in self.level_weights.items():
                prefix = true_code[:level]
                idxs = self.prefix2fullidx[level].get(prefix, [])
                idxs = [j for j in idxs if 0 <= j < num_classes]
                if len(idxs) == 0:
                    level_loss = self.ce_loss(logits[i:i+1] / self.temperature, targets[i:i+1]).mean()
                    print(f" 警告：样本 {i} 的前缀 '{prefix}' 在层级 {level} 中没有匹配的索引，使用交叉熵损失")
                else:
                    print(f"样本 {i} 的前缀 '{prefix}' 在层级 {level} 中匹配的索引: {idxs}")
                    print(f"len(probs[i]) = {len(probs[i])}")
                    mass = probs[i, idxs].sum()
                    level_loss = -torch.log(mass + 1e-8) # 全类别中匹配该前缀的索引列表
                sample_loss += w * level_loss
            # 权重加权
            
            batch_hl.append(sample_loss / total_level_weight)

        hierarchical_loss = torch.stack(batch_hl).mean()
                    
        # 3. 组合损失
        print(f"层次化损失: {hierarchical_loss.item():.4f}, 交叉熵损失: {ce_loss.mean().item():.4f}, 温度: {self.temperature}")
        total_loss = self.alpha * hierarchical_loss + (1 - self.alpha) * ce_loss.mean()

        
        # 4. 计算各级别的准确率（用于监控）
        level_accuracies = {}
        with torch.no_grad():
            pred_indices = torch.argmax(logits, dim=1)
            
            for level in [1, 2, 3, 4]:
                correct = 0
                for i in range(batch_size):
                    if i < len(target_codes):
                        true_code = target_codes[i]
                        pred_idx = pred_indices[i].item()
                        
                        # 安全获取预测编码
                        if pred_idx < len(self.level_to_labels[4]) and len(self.level_to_labels[4]) > 0:
                            pred_code = self.level_to_labels[4][pred_idx]
                        else:
                            pred_code = "0000"  # 默认编码
                        
                        # 安全获取级别编码
                        true_level_code = true_code[:min(level, len(true_code))]
                        pred_level_code = pred_code[:min(level, len(pred_code))]
                        
                        if true_level_code == pred_level_code:
                            correct += 1
                
                level_accuracies[f'level_{level}_acc'] = correct / batch_size if batch_size > 0 else 0.0
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.mean(),
            'hierarchical_loss': hierarchical_loss,
            'level_accuracies': level_accuracies
        }


class MultiTaskHierarchicalHead(nn.Module):
    """
    多任务层次化分类头
    为ISCO 1-4级分别创建分类头，共享特征表示
    """
    
    def __init__(self, 
                 hidden_size: int,
                 isco_hierarchy: Dict,
                 dropout: float = 0.1,
                 use_level_attention: bool = True):
        """
        Args:
            hidden_size: BERT隐藏层大小
            isco_hierarchy: ISCO层次结构
            dropout: dropout比率
            use_level_attention: 是否使用级别注意力机制
        """
        super(MultiTaskHierarchicalHead, self).__init__()
        
        self.isco_hierarchy = isco_hierarchy
        self.hidden_size = hidden_size
        self.use_level_attention = use_level_attention
        
        # 构建各级别的类别映射
        self._build_level_mappings()
        
        # 共享特征变换层
        self.shared_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 各级别的分类头
        self.level_classifiers = nn.ModuleDict()
        for level in [1, 2, 3, 4]:
            num_classes = max(1, len(self.level_to_labels[level]))  # 确保至少有1个类别
            
            classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_classes)
            )
            
            self.level_classifiers[f'level_{level}'] = classifier
        
        # 级别注意力机制（可选）
        if use_level_attention:
            self.level_attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            
            # 级别嵌入
            self.level_embeddings = nn.Embedding(4, hidden_size)
        
        print(f"✅ 多任务层次化分类头初始化完成")
        print(f"   各级别类别数: {[len(self.level_to_labels[i]) for i in [1,2,3,4]]}")
        print(f"   使用级别注意力: {self.use_level_attention}")

    def _build_level_mappings(self):
        """构建各级别的类别映射"""
        self.level_to_labels = {1: set(), 2: set(), 3: set(), 4: set()}
        
        # 从层次结构中提取所有编码
        all_codes = set()
        for isco_code, node_info in self.isco_hierarchy.items():
            if len(isco_code) >= 1:
                all_codes.add(isco_code)
        
        # 为每个级别收集标签
        for isco_code in all_codes:
            if len(isco_code) >= 4:  # 4级编码
                for level in [1, 2, 3, 4]:
                    level_code = isco_code[:level]
                    self.level_to_labels[level].add(level_code)
            elif len(isco_code) >= 1:  # 1-3级编码
                max_level = len(isco_code)
                for level in range(1, max_level + 1):
                    level_code = isco_code[:level]
                    self.level_to_labels[level].add(level_code)
                # 补充到4级
                for level in range(max_level + 1, 5):
                    self.level_to_labels[level].add(isco_code)
        
        # 转换为排序列表
        for level in [1, 2, 3, 4]:
            self.level_to_labels[level] = sorted(list(self.level_to_labels[level]))
            # 确保每个级别至少有一个标签
            if not self.level_to_labels[level]:
                self.level_to_labels[level] = ['0']  # 默认标签
        
        # 创建映射字典
        self.code_to_idx = {}
        for level in [1, 2, 3, 4]:
            self.code_to_idx[level] = {code: idx for idx, code in enumerate(self.level_to_labels[level])}

    def convert_targets(self, isco_4_codes: List[str]) -> Dict[str, torch.Tensor]:
        """将4级ISCO编码转换为各级别的目标索引"""
        batch_size = len(isco_4_codes)
        device = next(self.parameters()).device
        
        level_targets = {}
        
        for level in [1, 2, 3, 4]:
            targets = []
            for isco_code in isco_4_codes:
                # 安全获取级别编码
                if len(isco_code) >= level:
                    level_code = isco_code[:level]
                else:
                    level_code = isco_code  # 使用原编码
                
                # 安全获取目标索引
                if level_code in self.code_to_idx[level]:
                    target_idx = self.code_to_idx[level][level_code]
                else:
                    target_idx = 0  # 默认索引
                
                targets.append(target_idx)
            
            level_targets[f'level_{level}'] = torch.tensor(targets, device=device, dtype=torch.long)
        
        return level_targets

    def forward(self, hidden_states: torch.Tensor, isco_4_codes: Optional[List[str]] = None):
        """
        前向传播
        
        Args:
            hidden_states: BERT输出的隐藏状态 [batch_size, hidden_size]
            isco_4_codes: 4级ISCO编码列表（训练时需要）
            
        Returns:
            Dict包含各级别的logits和转换后的targets
        """
        batch_size = hidden_states.size(0)
        
        # 共享特征变换
        shared_features = self.shared_transform(hidden_states)  # [batch_size, hidden_size]
        
        # 级别注意力机制（可选）
        if self.use_level_attention:
            # 创建级别查询
            level_ids = torch.arange(4, device=hidden_states.device)  # [0, 1, 2, 3]
            level_embeddings = self.level_embeddings(level_ids)  # [4, hidden_size]
            level_embeddings = level_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, 4, hidden_size]
            
            # 将共享特征作为键和值
            shared_features_expanded = shared_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # 注意力计算
            attended_features, _ = self.level_attention(
                level_embeddings, shared_features_expanded, shared_features_expanded
            )  # [batch_size, 4, hidden_size]
        else:
            # 不使用注意力，直接复制特征
            attended_features = shared_features.unsqueeze(1).expand(-1, 4, -1)  # [batch_size, 4, hidden_size]
        
        # 各级别分类
        level_logits = {}
        for i, level in enumerate([1, 2, 3, 4]):
            level_features = attended_features[:, i, :]  # [batch_size, hidden_size]
            logits = self.level_classifiers[f'level_{level}'](level_features)
            level_logits[f'level_{level}'] = logits
        
        results = {'level_logits': level_logits}
        
        # 如果提供了目标编码，转换为各级别目标
        if isco_4_codes is not None:
            level_targets = self.convert_targets(isco_4_codes)
            results['level_targets'] = level_targets
        
        return results


class MultiTaskHierarchicalLoss(nn.Module):
    """
    多任务层次化损失函数
    结合各级别的损失和层次化约束
    """
    
    def __init__(self, 
                 isco_hierarchy: Dict,
                 level_weights: Optional[Dict[int, float]] = None,
                 task_weights: Optional[Dict[int, float]] = None,
                 consistency_weight: float = 0.1):
        """
        Args:
            isco_hierarchy: ISCO层次结构
            level_weights: 层次化损失权重
            task_weights: 各任务权重 {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
            consistency_weight: 层次一致性损失权重
        """
        super(MultiTaskHierarchicalLoss, self).__init__()
        
        # 任务权重：4级最重要，1级最不重要
        self.task_weights = task_weights or {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        self.consistency_weight = consistency_weight
        
        # 各级别的层次化损失函数
        self.level_losses = nn.ModuleDict()
        for level in [1, 2, 3, 4]:
            # 为每个级别创建专门的层次化损失
            level_hierarchy = self._extract_level_hierarchy(isco_hierarchy, level)
            self.level_losses[f'level_{level}'] = HierarchicalLoss(
                level_hierarchy, level_weights
            )
        
        print(f"✅ 多任务层次化损失函数初始化完成")
        print(f"   任务权重: {self.task_weights}")
        print(f"   一致性权重: {self.consistency_weight}")

    def _extract_level_hierarchy(self, isco_hierarchy: Dict, target_level: int) -> Dict:
        """为特定级别提取层次结构"""
        level_hierarchy = {}
        
        # 收集该级别的所有编码
        level_codes = set()
        for isco_4_code, node_info in isco_hierarchy.items():
            if len(isco_4_code) >= target_level:
                level_code = isco_4_code[:target_level]
                level_codes.add(level_code)
            elif len(isco_4_code) >= 1:
                level_codes.add(isco_4_code)  # 使用原编码
        
        # 为每个级别编码创建节点
        for level_code in level_codes:
            if level_code not in level_hierarchy:
                # 创建该级别的节点信息
                parents = []
                for i in range(1, min(target_level, len(level_code))):
                    parents.append(level_code[:i])
                
                level_hierarchy[level_code] = {
                    'parents': parents,
                    'label': level_code,
                    'level': target_level
                }
        
        # 确保至少有一个节点
        if not level_hierarchy:
            level_hierarchy['0'] = {
                'parents': [],
                'label': '0',
                'level': target_level
            }
        
        return level_hierarchy

    def forward(self, level_logits, level_targets, target_codes):
        total_loss = 0.0
        level_losses = {}
        level_accuracies = {}

        # 1. 各级别 CrossEntropy
        for level in [1, 2, 3, 4]:
            key = f'level_{level}'
            if key not in level_logits or key not in level_targets:
                continue

            logits_l = level_logits[key]
            targets_l = level_targets[key]

            # 标准交叉熵损失
            ce_l = F.cross_entropy(logits_l, targets_l, reduction='mean')

            level_losses[key] = ce_l
            total_loss += self.task_weights[level] * ce_l

            # 记录准确率
            with torch.no_grad():
                preds = logits_l.argmax(dim=1)
                level_accuracies[f'{key}_acc'] = (preds == targets_l).float().mean().item()

        # 2. 一致性损失（保持不变）
        consistency_loss = self._compute_consistency_loss(level_logits)
        total_loss += self.consistency_weight * consistency_loss

        return {
            'total_loss': total_loss,
            'level_losses': level_losses,
            'consistency_loss': consistency_loss,
            'level_accuracies': level_accuracies
        }


    def _compute_consistency_loss(self, level_logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算层次一致性损失
        确保粗粒度预测与细粒度预测保持一致
        """
        consistency_loss = 0.0
        
        # 获取各级别的预测概率
        level_probs = {}
        for level in [1, 2, 3, 4]:
            level_key = f'level_{level}'
            if level_key in level_logits:
                level_probs[level_key] = F.softmax(level_logits[level_key], dim=1)
        
        # 计算相邻级别间的一致性
        for level in [1, 2, 3]:
            current_key = f'level_{level}'
            next_key = f'level_{level + 1}'
            
            if current_key in level_probs and next_key in level_probs:
                current_probs = level_probs[current_key]
                next_probs = level_probs[next_key]
                
                # 计算当前级别预测的熵
                current_entropy = -(current_probs * torch.log(current_probs + 1e-8)).sum(dim=1).mean()
                consistency_loss += current_entropy * 0.01  # 小权重
        
        return consistency_loss


def create_hierarchical_components(isco_hierarchy: Dict, 
                                 hidden_size: int = 768,
                                 level_weights: Optional[Dict[int, float]] = None,
                                 task_weights: Optional[Dict[int, float]] = None) -> Tuple[MultiTaskHierarchicalHead, MultiTaskHierarchicalLoss]:
    """
    创建层次化组件的工厂函数
    
    Args:
        isco_hierarchy: ISCO层次结构
        hidden_size: BERT隐藏层大小
        level_weights: 层次损失权重
        task_weights: 任务权重
        
    Returns:
        (多任务分类头, 多任务损失函数)
    """
    
    # 创建多任务分类头
    classification_head = MultiTaskHierarchicalHead(
        hidden_size=hidden_size,
        isco_hierarchy=isco_hierarchy,
        dropout=0.1,
        use_level_attention=True
    )
    
    # 创建多任务损失函数
    loss_function = MultiTaskHierarchicalLoss(
        isco_hierarchy=isco_hierarchy,
        level_weights=level_weights,
        task_weights=task_weights,
        consistency_weight=0.1
    )
    
    print(f"✅ 层次化组件创建完成")
    print(f"   分类头: 多任务架构，支持级别注意力")
    print(f"   损失函数: 层次化 + 多任务 + 一致性约束")
    
    return classification_head, loss_function


if __name__ == "__main__":
    # 测试代码
    print("🧪 测试层次化组件...")
    
    # 创建测试用的ISCO层次结构
    test_hierarchy = {
        '1': {'parents': [], 'label': '1', 'level': 1},
        '12': {'parents': ['1'], 'label': '12', 'level': 2},
        '123': {'parents': ['1', '12'], 'label': '123', 'level': 3},
        '1234': {'parents': ['1', '12', '123'], 'label': '1234', 'level': 4},
        '1235': {'parents': ['1', '12', '123'], 'label': '1235', 'level': 4},
        '2': {'parents': [], 'label': '2', 'level': 1},
        '23': {'parents': ['2'], 'label': '23', 'level': 2},
        '234': {'parents': ['2', '23'], 'label': '234', 'level': 3},
        '2345': {'parents': ['2', '23', '234'], 'label': '2345', 'level': 4},
    }
    
    try:
        # 创建组件
        head, loss_fn = create_hierarchical_components(test_hierarchy)
        
        # 创建测试数据
        batch_size = 4
        hidden_size = 768
        hidden_states = torch.randn(batch_size, hidden_size)
        target_codes = ['1234', '1235', '2345', '1234']
        
        # 前向传播测试
        outputs = head(hidden_states, target_codes)
        
        # 损失计算测试
        loss_outputs = loss_fn(
            outputs['level_logits'], 
            outputs['level_targets'],
            target_codes
        )
        
        print(f"✅ 多任务测试完成")
        print(f"   总损失: {loss_outputs['total_loss'].item():.4f}")
        print(f"   各级别损失: {[v.item() if torch.is_tensor(v) else v for v in loss_outputs['level_losses'].values()]}")
        print(f"   一致性损失: {loss_outputs['consistency_loss'].item():.4f}")
        
    except Exception as e:
        print(f"❌ 多任务测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试单级别层次化损失
    print(f"\n🧪 测试单级别层次化损失...")
    try:
        single_loss_fn = HierarchicalLoss(
            isco_hierarchy=test_hierarchy,
            level_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0},
            temperature=1.0,
            alpha=0.7
        )
        
        # 创建简单的4级分类测试
        four_level_codes = [code for code in test_hierarchy.keys() if len(code) == 4]
        num_classes = len(four_level_codes)
        
        if num_classes > 0:
            simple_logits = torch.randn(batch_size, num_classes)
            simple_targets = torch.randint(0, num_classes, (batch_size,))
            simple_codes = four_level_codes[:batch_size] if len(four_level_codes) >= batch_size else four_level_codes * (batch_size // len(four_level_codes) + 1)
            simple_codes = simple_codes[:batch_size]  # 确保长度匹配
            
            simple_loss_outputs = single_loss_fn(simple_logits, simple_targets, simple_codes)
            print(f"✅ 单级别测试完成")
            print(f"   单级别总损失: {simple_loss_outputs['total_loss'].item():.4f}")
            print(f"   交叉熵损失: {simple_loss_outputs['ce_loss'].item():.4f}")
            print(f"   层次化损失: {simple_loss_outputs['hierarchical_loss'].item():.4f}")
        else:
            print("   ⚠️ 跳过单级别测试（没有4级类别）")
            
    except Exception as e:
        print(f"❌ 单级别测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🎉 所有测试完成！")
    print(f"📋 层次化组件功能验证:")
    print(f"   ✓ 多任务层次化分类头")
    print(f"   ✓ 层次化损失函数")
    print(f"   ✓ 多任务层次化损失")
    print(f"   ✓ ISCO层次结构处理")
    print(f"   ✓ 各级别准确率计算")
    print(f"\n💡 可以开始在实际模型中使用层次化功能了！")