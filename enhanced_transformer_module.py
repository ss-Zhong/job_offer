#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版Transformer模块
集成层次化损失函数和多任务学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# 导入基础模块
from job_offers_classifier.transformer_module import TransformerClassifier

# 导入层次化损失函数
from hierarchical_loss import HierarchicalISCOLoss, HierarchicalMultitaskLoss


class EnhancedTextDataset(torch.utils.data.Dataset):
    """增强版文本数据集，支持ISCO编码"""
    
    def __init__(self, texts, labels=None, isco_codes=None, num_labels=None, 
                 lazy_encode=True, labels_dense_vec=False):
        self.texts = texts
        self.labels = labels
        self.isco_codes = isco_codes  # 原始ISCO编码字符串
        self.num_labels = num_labels
        self.lazy_encode = lazy_encode
        self.labels_dense_vec = labels_dense_vec
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {'text': self.texts[idx]}
        
        if self.labels is not None:
            if self.labels_dense_vec:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            else:
                item['labels'] = self.labels[idx]
                
        if self.isco_codes is not None:
            item['isco_code'] = self.isco_codes[idx]
            
        return item


class HierarchicalClassificationHead(nn.Module):
    """层次化分类头，支持多级别输出"""
    
    def __init__(self, 
                 hidden_size: int,
                 hierarchy: Dict,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.hierarchy = hierarchy
        self.dropout = nn.Dropout(dropout_rate)
        
        # 构建各级别的映射
        self._build_level_mappings()
        
        # 为每个级别创建分类头
        self.level_classifiers = nn.ModuleDict()
        
        for level in [1, 2, 3, 4]:
            num_classes = self.level_info[level]['num_classes']
            self.level_classifiers[str(level)] = nn.Linear(hidden_size, num_classes)
        
        print(f"✅ 层次化分类头初始化完成")
        for level in [1, 2, 3, 4]:
            print(f"   Level {level}: {self.level_info[level]['num_classes']} classes")

    def _build_level_mappings(self):
        """构建各级别的类别信息"""
        self.level_info = {}
        
        # 获取所有4级编码
        level_4_codes = sorted([
            code for code, info in self.hierarchy.items() 
            if info['level'] == 4
        ])
        
        # 为每个级别构建类别集合
        for level in [1, 2, 3, 4]:
            level_codes = sorted(list(set([
                code[:level] for code in level_4_codes
            ])))
            
            self.level_info[level] = {
                'codes': level_codes,
                'code_to_idx': {code: idx for idx, code in enumerate(level_codes)},
                'idx_to_code': {idx: code for idx, code in enumerate(level_codes)},
                'num_classes': len(level_codes)
            }

    def forward(self, hidden_states: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        前向传播
        
        Args:
            hidden_states: BERT输出的隐藏状态 [batch_size, hidden_size]
            
        Returns:
            logits_dict: 各级别的logits {1: tensor, 2: tensor, 3: tensor, 4: tensor}
        """
        # 应用dropout
        hidden_states = self.dropout(hidden_states)
        
        # 计算各级别的logits
        logits_dict = {}
        for level in [1, 2, 3, 4]:
            logits_dict[level] = self.level_classifiers[str(level)](hidden_states)
        
        return logits_dict


class EnhancedTransformerClassifier(TransformerClassifier):
    """
    增强版Transformer分类器
    支持层次化损失和多任务学习
    """
    
    def __init__(self,
                 model_name_or_path: str,
                 output_size: int,
                 isco_hierarchy: Dict = None,
                 use_hierarchical_loss: bool = True,
                 use_multitask_learning: bool = False,
                 hierarchical_loss_weights: Dict[int, float] = None,
                 task_weights: Dict[int, float] = None,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.01,
                 train_batch_size: int = 16,
                 eval_batch_size: int = 16,
                 accumulate_grad_batches: int = 1,
                 max_seq_length: int = 512,
                 verbose: bool = True,
                 **kwargs):
        
        # 初始化父类
        super().__init__(
            model_name_or_path=model_name_or_path,
            output_size=output_size,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            max_seq_length=max_seq_length,
            verbose=verbose,
            **kwargs
        )
        
        # 保存层次化配置
        self.isco_hierarchy = isco_hierarchy
        
        self.use_hierarchical_loss = use_hierarchical_loss and isco_hierarchy is not None
        self.use_multitask_learning = use_multitask_learning and isco_hierarchy is not None
        
        # 损失权重
        self.hierarchical_loss_weights = hierarchical_loss_weights
        self.task_weights = task_weights
        
        # 替换输出层
        if self.use_multitask_learning:
            # 移除原有的输出层
            self.fc_head = None
            # 创建层次化分类头
            self.hierarchical_head = HierarchicalClassificationHead(
                hidden_size=self.transformer.config.hidden_size,
                hierarchy=self.isco_hierarchy,
                dropout_rate=0.1
            )
            
        # 创建损失函数
        if self.use_multitask_learning:
            self.criterion = HierarchicalMultitaskLoss(
                hierarchy=self.isco_hierarchy,
                task_weights=self.task_weights,
                level_weights=self.hierarchical_loss_weights
            )
        elif self.use_hierarchical_loss:
            self.criterion = HierarchicalISCOLoss(
                hierarchy=self.isco_hierarchy,
                level_weights=self.hierarchical_loss_weights
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # 用于记录各级别的准确率
        self.level_correct = {1: 0, 2: 0, 3: 0, 4: 0}
        self.level_total = {1: 0, 2: 0, 3: 0, 4: 0}
        
        if self.verbose:
            print(f"✅ 增强版Transformer分类器初始化完成")
            print(f"   层次化损失: {'启用' if self.use_hierarchical_loss else '禁用'}")
            print(f"   多任务学习: {'启用' if self.use_multitask_learning else '禁用'}")

    # def forward(self, input_ids, attention_mask, token_type_ids=None):
    #     """前向传播"""
    #     # 获取BERT输出
    #     outputs = self.transformer(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids
    #     )
        
    #     # 使用CLS token的表示
    #     pooled_output = outputs.last_hidden_state[:, 0]
        
    #     if self.use_multitask_learning:
    #         # 多任务输出
    #         logits_dict = self.hierarchical_head(pooled_output)
    #         return logits_dict
    #     else:
    #         # 单任务输出
    #         pooled_output = self.dropout(pooled_output)
    #         logits = self.fc_head(pooled_output)
    #         return logits

    def forward(self, batch, labels=None):
        
        transformer_output = self.transformer(batch['input_ids'], attention_mask=batch['attention_mask'])
        # 使用[CLS] token的输出
        transformer_output = transformer_output.last_hidden_state[:, 0, :]
        
        if self.use_multitask_learning:
            return self.hierarchical_head(transformer_output)
        elif self.fc_head is not None:
            output_ = self.fc_head(transformer_output, labels=labels)
            return output_
        else:
            print("Fault in EnhancedTransformerClassifier Forward")
            return transformer_output

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].to(self.device)
        
        # 前向传播
        outputs = self.forward(batch)
        
        # 计算损失
        if self.use_multitask_learning:
            loss, loss_dict = self.criterion(outputs, batch['labels'])
            
            # 记录各级别损失
            for level in [1, 2, 3, 4]:
                if f'level_{level}_loss' in loss_dict:
                    self.log(f'train_level_{level}_loss', loss_dict[f'level_{level}_loss'], 
                            on_step=True, on_epoch=True, prog_bar=False)
        else:
            loss = self.criterion(outputs, batch['labels'])
        
        # 记录总损失
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        # 计算损失和准确率
        if self.use_multitask_learning:
            # # 这里未实现
            # pass
            
            # if batch['labels'] is not None:  # Windows fix
            #     batch['labels'] = batch['labels'].type(torch.LongTensor)
            #     batch['labels'] = batch['labels'].to(self.device)
            # labels = batch['labels']
        
            # # 前向传播
            # outputs = self.forward(batch)
            
            # loss, loss_dict = self.criterion(outputs, labels)
            
            # # 计算各级别准确率
            # self._calculate_hierarchical_accuracy(outputs, labels)
            
            # # 记录损失
            # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # predictions = torch.argmax(outputs[4], dim=1)
            # return loss
        
            if batch['labels'] is not None:  # Windows fix
                batch['labels'] = batch['labels'].type(torch.LongTensor)
                batch['labels'] = batch['labels'].to(self.device)

            outputs = self.forward(batch)
            loss, loss_dict = self.criterion(outputs, batch['labels'])
            scores = F.softmax(outputs[4], dim=-1)
            
            if self.metrics is None:
                return loss
            
            # 处理层次化预测
            if self.hparams.labels_paths is not None:
                new_scores = torch.zeros((scores.shape[0], self.eval_num_labels)).to(self.device)
                for i, path in enumerate(self.hparams.labels_paths):
                    path_score = scores[:, path]
                    new_scores[:, i] = path_score.prod(axis=1)
                scores = new_scores
                assert scores.shape == (scores.shape[0], self.eval_num_labels)
                true_loss = F.cross_entropy(scores, batch['labels'])
                self.log(f'val_true_loss', true_loss, on_epoch=True, logger=True)

            # 修复：分别记录每个指标，而不是记录整个字典
            try:
                metrics_dict = self.metrics(scores, batch['labels'])
                for metric_name, metric_value in metrics_dict.items():
                    self.log(f'val_{metric_name}', metric_value, on_epoch=True, logger=True)
            except Exception as e:
                # 如果指标计算失败，只记录损失
                if self.hparams.verbose:
                    print(f"Warning: Metrics calculation failed: {e}")
            
            self.log(f'val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
            
            return loss
        
        else:
            return self._eval_step(batch, eval_name='val')

    def validation_epoch_end(self, outputs):
        """验证epoch结束时的处理"""
        if self.use_multitask_learning and sum(self.level_total.values()) > 0:
            # 计算并记录各级别准确率
            for level in [1, 2, 3, 4]:
                if self.level_total[level] > 0:
                    accuracy = self.level_correct[level] / self.level_total[level]
                    self.log(f'val_level_{level}_accuracy', accuracy, on_epoch=True)
            
            # 重置计数器
            self.level_correct = {1: 0, 2: 0, 3: 0, 4: 0}
            self.level_total = {1: 0, 2: 0, 3: 0, 4: 0}

    def _calculate_hierarchical_accuracy(self, logits_dict: Dict[int, torch.Tensor], 
                                        targets: torch.Tensor):
        """计算各级别的准确率"""
        if not hasattr(self.criterion, 'hierarchical_loss'):
            return
        
        # 获取目标ISCO编码
        target_codes = []
        for idx in targets:
            if hasattr(self.criterion.hierarchical_loss, 'idx_to_code'):
                target_codes.append(self.criterion.hierarchical_loss.idx_to_code[idx.item()])
            else:
                # 如果没有映射，跳过
                return
        
        # 计算各级别准确率
        for level in [1, 2, 3, 4]:
            if level in logits_dict:
                predictions = torch.argmax(logits_dict[level], dim=1)
                
                for i, pred_idx in enumerate(predictions):
                    # 获取预测的编码
                    if hasattr(self.hierarchical_head, 'level_info'):
                        pred_code = self.hierarchical_head.level_info[level]['idx_to_code'][pred_idx.item()]
                        true_code = target_codes[i][:level]
                        
                        self.level_total[level] += 1
                        if pred_code == true_code:
                            self.level_correct[level] += 1

    def predict_step(self, batch, batch_idx):
        """预测步骤"""
        # input_ids = batch['input_ids']
        # attention_mask = batch['attention_mask']
        # token_type_ids = batch.get('token_type_ids', None)
        
        # 前向传播
        outputs = self.forward(batch)
        
        if self.use_multitask_learning:
            # 返回4级预测的概率
            logits = outputs[4]
        else:
            if isinstance(outputs, dict):
                logits = outputs[4]
            else:
                logits = outputs
        
        # 返回softmax概率
        probs = F.softmax(logits, dim=-1)
        return probs

    def configure_optimizers(self):
        """配置优化器，支持不同学习率"""
        # 分组参数
        no_decay = ['bias', 'LayerNorm.weight']
        
        # 基础参数组
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.transformer.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.hparams.weight_decay,
                'lr': self.hparams.learning_rate
            },
            {
                'params': [p for n, p in self.transformer.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.hparams.learning_rate
            }
        ]
        
        # 如果使用多任务学习，为分类头添加更高的学习率
        if self.use_multitask_learning:
            classifier_params = []
            for level_classifier in self.hierarchical_head.level_classifiers.values():
                classifier_params.extend(list(level_classifier.parameters()))
            
            optimizer_grouped_parameters.append({
                'params': classifier_params,
                'weight_decay': self.hparams.weight_decay,
                'lr': self.hparams.learning_rate * 2  # 分类头使用2倍学习率
            })
        elif hasattr(self, 'fc_head') and self.fc_head is not None:
            optimizer_grouped_parameters.append({
                'params': self.fc_head.parameters(),
                'weight_decay': self.hparams.weight_decay,
                'lr': self.hparams.learning_rate * 2
            })
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            eps=self.hparams.adam_epsilon
        )
        
        # 学习率调度器
        scheduler = self._get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def _get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """线性学习率调度器"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / 
                float(max(1, num_training_steps - num_warmup_steps))
            )
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def get_hierarchical_predictions(self, texts: List[str], top_k: int = 5) -> Dict:
        """
        获取层次化预测结果
        
        Args:
            texts: 输入文本列表
            top_k: 每个级别返回的top-k预测
            
        Returns:
            predictions: 包含各级别预测的字典
        """
        self.eval()
        
        # 准备数据
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.model_name_or_path)
        
        all_predictions = {1: [], 2: [], 3: [], 4: []}
        
        with torch.no_grad():
            for text in texts:
                # 编码文本
                inputs = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=self.hparams.max_seq_length,
                    return_tensors='pt'
                )
                
                # 移到正确的设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 前向传播
                outputs = self.forward(**inputs)
                
                if self.use_multitask_learning:
                    # 处理多任务输出
                    for level in [1, 2, 3, 4]:
                        if level in outputs:
                            probs = F.softmax(outputs[level], dim=-1)
                            top_k_probs, top_k_indices = torch.topk(probs[0], k=min(top_k, probs.shape[1]))
                            
                            # 转换为编码
                            predictions = []
                            for idx, prob in zip(top_k_indices, top_k_probs):
                                code = self.hierarchical_head.level_info[level]['idx_to_code'][idx.item()]
                                predictions.append({
                                    'code': code,
                                    'probability': prob.item()
                                })
                            
                            all_predictions[level].append(predictions)
                else:
                    # 单任务输出，只有4级预测
                    probs = F.softmax(outputs, dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs[0], k=min(top_k, probs.shape[1]))
                    
                    predictions = []
                    for idx, prob in zip(top_k_indices, top_k_probs):
                        predictions.append({
                            'index': idx.item(),
                            'probability': prob.item()
                        })
                    
                    all_predictions[4].append(predictions)
        
        return all_predictions


# 集成到主分类器的辅助函数
def integrate_hierarchical_loss(classifier, hierarchy, use_hierarchical=True, use_multitask=False):
    """
    将层次化损失集成到现有分类器
    
    Args:
        classifier: 现有的分类器实例
        hierarchy: ISCO层次结构
        use_hierarchical: 是否使用层次化损失
        use_multitask: 是否使用多任务学习
    """
    if use_multitask:
        # 替换为多任务损失
        classifier.criterion = HierarchicalMultitaskLoss(
            hierarchy=hierarchy,
            task_weights={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
            level_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
        )
        print("✅ 已启用多任务层次化学习")
    elif use_hierarchical:
        # 替换为层次化损失
        classifier.criterion = HierarchicalISCOLoss(
            hierarchy=hierarchy,
            level_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
        )
        print("✅ 已启用层次化损失函数")
    
    return classifier


# 使用示例
if __name__ == "__main__":
    print("🚀 增强版Transformer分类器模块")
    print("=" * 50)
    
    # 示例：创建增强版分类器
    print("\n创建示例分类器...")
    
    # 模拟ISCO层次结构
    example_hierarchy = {
        '1': {'label': '1', 'level': 1, 'parents': []},
        '11': {'label': '11', 'level': 2, 'parents': ['1']},
        '112': {'label': '112', 'level': 3, 'parents': ['1', '11']},
        '1121': {'label': '1121', 'level': 4, 'parents': ['1', '11', '112']},
        # ... 更多编码
    }
    
    # 创建分类器
    classifier = EnhancedTransformerClassifier(
        model_name_or_path='bert-base-chinese',
        output_size=100,  # 假设有100个4级类别
        isco_hierarchy=example_hierarchy,
        use_hierarchical_loss=True,
        use_multitask_learning=True,
        hierarchical_loss_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0},
        task_weights={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    )
    
    print("\n✅ 增强版分类器创建成功!")
    print("\n特性:")
    print("- 层次化损失函数：根据错误级别给予不同惩罚")
    print("- 多任务学习：同时预测1-4级ISCO编码")
    print("- 级别权重：1级错误惩罚最重(8.0)，4级最轻(1.0)")
    print("- 任务权重：4级预测权重最高(0.4)，1级最低(0.1)")