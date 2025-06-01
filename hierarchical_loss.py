#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层次化损失函数模块
为ISCO编码分类任务设计的多级别惩罚损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class HierarchicalISCOLoss(nn.Module):
    """
    层次化ISCO损失函数
    根据ISCO编码的层次结构，对不同级别的分类错误给予不同权重的惩罚
    """
    
    def __init__(self, 
                 hierarchy: Dict,
                 level_weights: Dict[int, float] = None,
                 base_loss_fn: str = 'cross_entropy',
                 reduction: str = 'mean'):
        """
        Args:
            hierarchy: ISCO层次结构字典
            level_weights: 各级别错误的惩罚权重
            base_loss_fn: 基础损失函数类型 ('cross_entropy' or 'focal')
            reduction: 损失聚合方式 ('mean' or 'sum')
        """
        super().__init__()
        
        self.hierarchy = hierarchy
        self.reduction = reduction
        
        # 默认权重：级别越高（数字越小），错误惩罚越大
        self.level_weights = level_weights
        
        # 构建ISCO编码映射
        self._build_isco_mappings()
        
        # 选择基础损失函数
        if base_loss_fn == 'cross_entropy':
            self.base_loss = nn.CrossEntropyLoss(reduction='none')
        elif base_loss_fn == 'focal':
            self.base_loss = FocalLoss(reduction='none')
        else:
            raise ValueError(f"Unknown base loss function: {base_loss_fn}")
        
        print(f"✅ 层次化损失函数初始化完成")
        print(f"   级别权重: {self.level_weights}")
        print(f"   基础损失: {base_loss_fn}")

    def _build_isco_mappings(self):
        """构建ISCO编码映射关系"""
        # 提取所有4位ISCO编码
        self.isco_codes = []
        self.code_to_idx = {}
        self.idx_to_code = {}
        
        # 只处理4位编码（最底层）
        level_4_codes = sorted([
            code for code, info in self.hierarchy.items() 
            if info['level'] == 4
        ])
        
        for idx, code in enumerate(level_4_codes):
            self.isco_codes.append(code)
            self.code_to_idx[code] = idx
            self.idx_to_code[idx] = code
        
        self.num_classes = len(self.isco_codes)
        
        # 构建层次距离矩阵
        self._build_hierarchy_distance_matrix()
        
        print(f"   构建了 {self.num_classes} 个ISCO-4级编码的映射")

    def _build_hierarchy_distance_matrix(self):
        """构建层次距离矩阵，用于计算不同编码之间的层次距离"""
        n = self.num_classes
        self.hierarchy_distance_matrix = torch.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                
                code_i = self.idx_to_code[i]
                code_j = self.idx_to_code[j]
                
                # 计算两个编码的层次距离
                distance = self._calculate_hierarchy_distance(code_i, code_j)
                self.hierarchy_distance_matrix[i, j] = distance

    def _calculate_hierarchy_distance(self, code1: str, code2: str) -> float:
        """
        计算两个ISCO编码之间的层次距离
        返回值基于最高不同级别的权重
        """
        # 从高到低检查每个级别
        for level in [1, 2, 3, 4]:
            if code1[:level] != code2[:level]:
                # 在这个级别上不同，返回对应权重
                return self.level_weights[level]
        
        # 完全相同（不应该发生）
        return 0.0

    def _get_hierarchy_weights(self, targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        根据预测和真实标签计算层次权重
        
        Args:
            targets: 真实标签索引 [batch_size]
            predictions: 预测的类别索引 [batch_size]
        
        Returns:
            weights: 每个样本的层次权重 [batch_size]
        """
        batch_size = targets.shape[0]
        weights = torch.ones(batch_size, device=targets.device)
        
        for i in range(batch_size):
            true_idx = targets[i].item()
            pred_idx = predictions[i].item()
            
            if true_idx != pred_idx:
                # 获取层次距离作为权重
                weights[i] = self.hierarchy_distance_matrix[true_idx, pred_idx].item()
        
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算层次化损失
        
        Args:
            logits: 模型输出的logits [batch_size, num_classes]
            targets: 真实标签索引 [batch_size]
        
        Returns:
            loss: 层次化加权损失
        """
        # 计算基础损失
        base_loss = self.base_loss(logits, targets)
        
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)
        
        # 计算层次权重
        hierarchy_weights = self._get_hierarchy_weights(targets, predictions)
        
        # 应用层次权重
        weighted_loss = base_loss * hierarchy_weights
        
        # 聚合损失
        if self.reduction == 'mean':
            return weighted_loss.mean()
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            return weighted_loss
    
    def get_detailed_loss_info(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict:
        """获取详细的损失信息，用于分析和调试"""
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=1)
            
            # 统计各级别的错误
            level_errors = {1: 0, 2: 0, 3: 0, 4: 0}
            correct = 0
            
            for i in range(len(targets)):
                true_idx = targets[i].item()
                pred_idx = predictions[i].item()
                
                if true_idx == pred_idx:
                    correct += 1
                else:
                    true_code = self.idx_to_code[true_idx]
                    pred_code = self.idx_to_code[pred_idx]
                    
                    # 找出错误级别
                    for level in [1, 2, 3, 4]:
                        if true_code[:level] != pred_code[:level]:
                            level_errors[level] += 1
                            break
            
            # 计算损失
            loss = self.forward(logits, targets)
            
            return {
                'total_loss': loss.item(),
                'accuracy': correct / len(targets),
                'level_1_errors': level_errors[1],
                'level_2_errors': level_errors[2],
                'level_3_errors': level_errors[3],
                'level_4_errors': level_errors[4],
                'total_errors': sum(level_errors.values())
            }


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class HierarchicalMultitaskLoss(nn.Module):
    """
    多任务层次化损失函数
    同时预测多个级别的ISCO编码
    """
    
    def __init__(self,
                 hierarchy: Dict,
                 task_weights: Dict[int, float] = None,
                 level_weights: Dict[int, float] = None):
        """
        Args:
            hierarchy: ISCO层次结构
            task_weights: 各任务（级别）的权重 {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
            level_weights: 各级别错误的惩罚权重
        """
        super().__init__()
        
        self.hierarchy = hierarchy
        
        # 任务权重（越精细的级别权重越高）
        self.task_weights = task_weights or {
            1: 0.1,  # 一级分类任务权重
            2: 0.2,  # 二级分类任务权重
            3: 0.3,  # 三级分类任务权重
            4: 0.4   # 四级分类任务权重（最重要）
        }
        
        # 构建各级别的类别映射
        self._build_level_mappings()
        
        # 为每个级别创建损失函数
        self.level_losses = nn.ModuleDict()
        for level in [1, 2, 3, 4]:
            self.level_losses[str(level)] = nn.CrossEntropyLoss()
        
        # 主要的层次化损失（用于4级分类）
        self.hierarchical_loss = HierarchicalISCOLoss(
            hierarchy=hierarchy,
            level_weights=level_weights
        )

    def _build_level_mappings(self):
        """构建各级别的编码映射"""
        self.level_mappings = {}
        
        for level in [1, 2, 3, 4]:
            level_codes = sorted(list(set([
                code[:level] for code, info in self.hierarchy.items()
                if info['level'] == 4  # 从4级编码提取
            ])))
            
            self.level_mappings[level] = {
                'codes': level_codes,
                'code_to_idx': {code: idx for idx, code in enumerate(level_codes)},
                'idx_to_code': {idx: code for idx, code in enumerate(level_codes)},
                'num_classes': len(level_codes)
            }

    def forward(self, 
                logits_dict: Dict[int, torch.Tensor], 
                targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        计算多任务层次化损失
        
        Args:
            logits_dict: 各级别的预测logits {1: [B, C1], 2: [B, C2], 3: [B, C3], 4: [B, C4]}
            targets: 4级ISCO编码的索引 [batch_size]
        
        Returns:
            total_loss: 总损失
            loss_dict: 各部分损失的详细信息
        """
        loss_dict = {}
        total_loss = 0
        
        # 获取各级别的真实标签
        target_codes = [self.hierarchical_loss.idx_to_code[idx.item()] for idx in targets]
        
        # 计算各级别的损失
        for level in [1, 2, 3, 4]:
            if level in logits_dict:
                # 获取该级别的目标
                level_targets = []
                for code in target_codes:
                    level_code = code[:level]
                    level_idx = self.level_mappings[level]['code_to_idx'][level_code]
                    level_targets.append(level_idx)
                
                level_targets = torch.tensor(level_targets, device=targets.device)
                
                # 计算该级别的损失
                if level == 4:
                    # 使用层次化损失
                    level_loss = self.hierarchical_loss(logits_dict[level], targets)
                else:
                    # 使用普通交叉熵
                    level_loss = self.level_losses[str(level)](logits_dict[level], level_targets)
                
                # 应用任务权重
                weighted_loss = self.task_weights[level] * level_loss
                
                loss_dict[f'level_{level}_loss'] = level_loss.item()
                total_loss += weighted_loss
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# 使用示例
def create_hierarchical_loss_example():
    """创建层次化损失函数的使用示例"""
    
    # 模拟层次结构
    hierarchy = {
        '1': {'label': '1', 'level': 1, 'parents': []},
        '11': {'label': '11', 'level': 2, 'parents': ['1']},
        '112': {'label': '112', 'level': 3, 'parents': ['1', '11']},
        '1121': {'label': '1121', 'level': 4, 'parents': ['1', '11', '112']},
        '1122': {'label': '1122', 'level': 4, 'parents': ['1', '11', '112']},
        '2': {'label': '2', 'level': 1, 'parents': []},
        '21': {'label': '21', 'level': 2, 'parents': ['2']},
        '212': {'label': '212', 'level': 3, 'parents': ['2', '21']},
        '2121': {'label': '2121', 'level': 4, 'parents': ['2', '21', '212']}
    }
    
    # 创建损失函数
    loss_fn = HierarchicalISCOLoss(
        hierarchy=hierarchy,
        level_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
    )
    
    # 模拟数据
    batch_size = 3
    num_classes = 3  # 简化示例
    
    # 模拟logits
    logits = torch.randn(batch_size, num_classes)
    
    # 模拟目标（使用索引）
    targets = torch.tensor([0, 1, 2])  # 对应 '1121', '1122', '2121'
    
    # 计算损失
    loss = loss_fn(logits, targets)
    
    # 获取详细信息
    info = loss_fn.get_detailed_loss_info(logits, targets)
    
    print(f"损失值: {loss.item():.4f}")
    print(f"详细信息: {info}")
    
    return loss_fn


if __name__ == "__main__":
    print("🎯 层次化损失函数模块")
    print("=" * 50)
    
    # 创建示例
    loss_fn = create_hierarchical_loss_example()
    
    print("\n💡 使用说明:")
    print("1. HierarchicalISCOLoss: 单任务层次化损失")
    print("   - 根据ISCO编码层级给予不同惩罚权重")
    print("   - 1级错误惩罚最重，4级错误惩罚最轻")
    print("\n2. HierarchicalMultitaskLoss: 多任务层次化损失")
    print("   - 同时预测多个级别的ISCO编码")
    print("   - 结合层次化惩罚和多任务学习")