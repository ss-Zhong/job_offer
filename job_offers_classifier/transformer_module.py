from typing import Optional, Dict, List
from pprint import pprint

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from pytorch_lightning import LightningModule

# 修复AdamW导入问题 - 新版本transformers中AdamW已移至torch
try:
    from transformers import AdamW
except ImportError:
    from torch.optim import AdamW

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from torchmetrics import MetricCollection, Recall, Precision, Accuracy, ConfusionMatrix

from job_offers_classifier.classification_utils import *


class FullyConnectedOutput(nn.Module):
    def __init__(self, input_size, output_size, layer_units=(10,), nonlin=nn.ReLU(), hidden_dropout=0, output_nonlin=nn.Softmax(dim=1), criterion=nn.CrossEntropyLoss(), labels_groups=None, labels_groups_mapping=None):
        super().__init__()
        self.input_size = input_size
        self.fc_head_size = output_size
        self.nonlin = nonlin
        self.layer_units = layer_units
        self.output_nonlin = output_nonlin
        self.criterion = criterion
        self.hidden_dropout = hidden_dropout
        self.labels_groups = labels_groups
        self.labels_groups_mapping = labels_groups_mapping

        if labels_groups_mapping is not None:
            self.register_buffer('_labels_groups_mapping', torch.tensor(labels_groups_mapping, dtype=torch.long))
            self.register_buffer('_labels_groups_mask_mapping', self._labels_groups_mapping != -1)
            self._labels_groups_mapping[~self._labels_groups_mask_mapping] += 1

        sequence = []
        units = [self.input_size] + list(self.layer_units) + [self.fc_head_size]
        for in_size, out_size in zip(units, units[1:]):
            sequence.extend([nn.Linear(in_size, out_size), self.nonlin, nn.Dropout(self.hidden_dropout)])

        sequence = sequence[:-2]
        self.sequential = nn.Sequential(*sequence)

    def forward(self, batch, labels=None):
        logits_output = self.sequential(batch)
            
        if self.labels_groups is None:
            output = self.output_nonlin(logits_output)
            # output = logits_output
            if labels is not None:
                return self.criterion(logits_output, labels).mean(), output
            else:
                return logits_output
            
        else:
            output = torch.zeros_like(logits_output, dtype=torch.float32)
            for group in self.labels_groups:
                output[:, group] = self.output_nonlin(logits_output[:, group])
        
            if labels is not None:
                labels_groups = self._labels_groups_mapping[labels]
                labels_groups_mask = self._labels_groups_mask_mapping[labels]
                criterion = 0
                for g_idx, group in enumerate(self.labels_groups):
                    assert (labels_groups[:, g_idx] < len(group)).all()
                    criterion += self.criterion(logits_output[:, group], labels_groups[:, g_idx]) * labels_groups_mask[:, g_idx]
                criterion = criterion.sum() / labels_groups_mask.sum()
                return criterion, output
            else:
                return logits_output


class TransformerClassifier(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        output_size: int,
        labels_groups: Optional[List] = None,
        labels_paths: Optional[List] = None,
        labels_groups_mapping: Optional[np.ndarray] = None,
        learning_rate: float = 1e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: float = 50,  # from 0-1 for % of training steps, >1 for number of steps
        weight_decay: float = 0.01,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        hidden_dropout: float = 0.0,
        eval_top_k: int = 10,
        freeze_transformer: bool = False,
        verbose: bool = True,
        # 新增中文适配参数
        language: str = 'auto',  # 'auto', 'zh', 'pl', 'en'
        chinese_model_optimization: bool = True,  # 是否自动优化中文模型参数
        **kwargs,
    ):
        super().__init__()

        # 自动检测语言和模型类型
        if language == 'auto':
            self.language = self._detect_language(model_name_or_path)
        else:
            self.language = language
            
        self.is_chinese_model = self._is_chinese_model(model_name_or_path)
        self.verbose = verbose
        
        # 为中文模型自动优化参数
        if self.is_chinese_model and chinese_model_optimization:
            learning_rate = self._optimize_chinese_learning_rate(learning_rate, model_name_or_path)
            warmup_steps = self._optimize_chinese_warmup_steps(warmup_steps)

        self.save_hyperparameters()

        self.eval_num_labels = output_size if labels_paths is None else len(labels_paths)

        if self.hparams.verbose:
            model_info = self._get_model_info(model_name_or_path)
            print(f"""Initializing TransformerClassifier:
                    Model: {model_name_or_path} ({model_info})
                    Language: {self.language}
                    Output size: {output_size}
                    Labels groups: {labels_groups is not None}
                    Learning rate: {learning_rate} (optimized: {self.is_chinese_model and chinese_model_optimization})
                    Weight decay: {weight_decay}
                    Warmup steps: {warmup_steps}
                    Batch sizes: train={train_batch_size}, eval={eval_batch_size}
                    Freeze transformer: {freeze_transformer}
                    Loss: {'hierarchical cross entropy' if labels_groups is not None else 'flat cross entropy'}
                    Hidden dropout: {hidden_dropout}
                    Hierarchy leaves: {self.eval_num_labels}""")

        # 加载模型配置和权重
        try:
            self.config = AutoConfig.from_pretrained(
                model_name_or_path,
                finetuning_task=None
            )
            self.transformer = AutoModel.from_pretrained(model_name_or_path, config=self.config)
            
            if self.hparams.verbose:
                print(f"✓ Successfully loaded model: {model_name_or_path}")
                print(f"  Hidden size: {self.config.hidden_size}")
                print(f"  Max position embeddings: {getattr(self.config, 'max_position_embeddings', 'N/A')}")
                
        except Exception as e:
            print(f"✗ Error loading model {model_name_or_path}: {e}")
            raise
            
        # 冻结transformer参数（如果需要）
        if freeze_transformer:
            for param in self.transformer.parameters():
                param.requires_grad = False
            if self.hparams.verbose:
                print("✓ Transformer parameters frozen")

        self.fc_head = FullyConnectedOutput(
            self.config.hidden_size, 
            output_size, 
            layer_units=(), 
            hidden_dropout=hidden_dropout, 
            output_nonlin=nn.Softmax(dim=1), 
            criterion=nn.CrossEntropyLoss(reduction='none'), 
            labels_groups=labels_groups, 
            labels_groups_mapping=labels_groups_mapping
        ) 
        
        # 设置评估指标
        metric_dict = {
            "acc/r@1": Accuracy(task="multiclass", num_classes=self.eval_num_labels),
            "macro_acc": Accuracy(task="multiclass", num_classes=self.eval_num_labels, average='macro'),
        }

        for i in range(2, min(self.eval_num_labels, eval_top_k + 1)):
            metric_dict[f"r@{i}"] = Recall(task="multiclass", num_classes=self.eval_num_labels, top_k=i)

        self.metrics = MetricCollection(metric_dict)

    def _detect_language(self, model_name_or_path: str) -> str:
        """自动检测模型语言"""
        model_lower = model_name_or_path.lower()
        if any(indicator in model_lower for indicator in ['chinese', 'hfl', 'ckiplab']):
            return 'zh'
        elif 'herbert' in model_lower or 'allegro' in model_lower:
            return 'pl'
        else:
            return 'en'
    
    def _is_chinese_model(self, model_name_or_path: str) -> bool:
        """检测是否为中文模型"""
        chinese_indicators = ['chinese', 'hfl', 'ckiplab', 'bert-base-chinese', 'macbert', 'roberta-wwm']
        return any(indicator in model_name_or_path.lower() for indicator in chinese_indicators)
    
    def _get_model_info(self, model_name_or_path: str) -> str:
        """获取模型信息描述"""
        model_lower = model_name_or_path.lower()
        if 'hfl' in model_lower:
            if 'roberta' in model_lower:
                return "HFL Chinese RoBERTa"
            elif 'macbert' in model_lower:
                return "HFL Chinese MacBERT"
            else:
                return "HFL Chinese BERT"
        elif 'chinese' in model_lower:
            return "Chinese BERT"
        elif 'herbert' in model_lower:
            return "Polish BERT"
        else:
            return "Multilingual/English BERT"
    
    def _optimize_chinese_learning_rate(self, current_lr: float, model_name: str) -> float:
        """为中文模型优化学习率"""
        if current_lr == 1e-5:  # 如果是默认值，则优化
            if 'roberta' in model_name.lower():
                return 2e-5  # RoBERTa通常需要稍高的学习率
            else:
                return 1.5e-5  # 其他中文BERT模型
        return current_lr
    
    def _optimize_chinese_warmup_steps(self, current_warmup: float) -> float:
        """为中文模型优化warmup步数"""
        if current_warmup == 50:  # 如果是默认值，则优化
            return 0.1  # 10%的训练步数
        return current_warmup

    def forward(self, batch, labels=None):
        transformer_output = self.transformer(batch['input_ids'], attention_mask=batch['attention_mask'])
        # 使用[CLS] token的输出
        transformer_output = transformer_output.last_hidden_state[:, 0, :]
        
        if self.fc_head is None:
            return transformer_output
        else:
            return self.fc_head.forward(transformer_output, labels=labels)
        
    def training_step(self, batch, batch_idx):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        
        # 记录训练指标
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def _eval_step(self, batch, eval_name='val'):
        if batch['labels'] is not None:  # Windows fix
            batch['labels'] = batch['labels'].type(torch.LongTensor)
            batch['labels'] = batch['labels'].to(self.device)
    
        loss, scores = self.forward(batch, batch['labels'])
        
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
            self.log(f'{eval_name}_true_loss', true_loss, on_epoch=True, logger=True)

        # 修复：分别记录每个指标，而不是记录整个字典
        try:
            metrics_dict = self.metrics(scores, batch['labels'])
            for metric_name, metric_value in metrics_dict.items():
                self.log(f'{eval_name}_{metric_name}', metric_value, on_epoch=True, logger=True)
        except Exception as e:
            # 如果指标计算失败，只记录损失
            if self.hparams.verbose:
                print(f"Warning: Metrics calculation failed: {e}")
        
        self.log(f'{eval_name}_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='val')

    def on_validation_epoch_end(self):
        """PyTorch Lightning 2.0+ 兼容"""
        try:
            if self.hparams.verbose:
                print("Validation performance:")
                metrics_result = self.metrics.compute()
                pprint(metrics_result)
            self.metrics.reset()
        except Exception as e:
            if self.hparams.verbose:
                print(f"Warning: Could not compute validation metrics: {e}")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._eval_step(batch, eval_name='test')

    def on_test_epoch_end(self):
        """PyTorch Lightning 2.0+ 兼容"""
        try:
            if self.hparams.verbose:
                print("Test performance:")
                pprint(self.metrics.compute())
        except Exception as e:
            if self.hparams.verbose:
                print(f"Warning: Could not compute test metrics: {e}")

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return

        # 获取dataloader计算总步数
        train_loader = self.trainer.datamodule.train_dataloader()

        # 计算总训练步数
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.num_devices)
        ab_size = self.trainer.accumulate_grad_batches
        dl_size = len(train_loader.dataset) * self.trainer.max_epochs
        self.total_steps = dl_size // tb_size // ab_size

        # 计算warmup步数
        self.num_warmup_steps = self.hparams.warmup_steps
        if self.hparams.warmup_steps < 1:
            self.num_warmup_steps = int(self.total_steps * self.hparams.warmup_steps)

        if self.hparams.verbose:
            print(f"Training setup:")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Warmup steps: {self.num_warmup_steps}")
            print(f"  Effective batch size: {tb_size}")
            print(f"  Dataset size: {len(train_loader.dataset)}")

    def configure_optimizers(self):
        # 准备优化器和调度器（线性warmup和衰减）
        optimizer = AdamW(
            self._get_optimizer_grouped_parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.total_steps,
        )
        
        scheduler = {
            "scheduler": scheduler, 
            "interval": "step", 
            "frequency": 1,
            "name": "linear_schedule_with_warmup"
        }
        
        return [optimizer], [scheduler]

    def save_transformer(self, ckpt_dir):
        """保存transformer模型"""
        self.transformer.save_pretrained(ckpt_dir)
        if self.hparams.verbose:
            print(f"✓ Transformer saved to {ckpt_dir}")

    def _get_optimizer_grouped_parameters(self, layer_wise_lr=False, layer_wise_lr_mutli=1.1):
        """获取优化器参数组"""
        # 建议不对bias、LayerNorm.weight使用权重衰减
        no_decay = ["bias", "LayerNorm.weight"]

        if layer_wise_lr:
            # 层级学习率（实验性功能）
            optimizer_grouped_parameters = []
            for name, params in self.named_parameters():
                weight_decay = 0.0 if any(nd in name for nd in no_decay) else self.hparams.weight_decay
                learning_rate = self.hparams.learning_rate

                if 'embeddings' in name or 'encoder' in name:
                    learning_rate /= 10

                    for i in range(0, 20):
                        if f'layer.{i}' in name:
                            learning_rate *= layer_wise_lr_mutli ** (i + 1)

                if self.hparams.verbose:
                    print(f"Layer-wise LR: {name} -> {learning_rate}")
                    
                optimizer_grouped_parameters.append({
                    "params": params,
                    "weight_decay": weight_decay,
                    "lr": learning_rate
                })

        else:
            # 标准优化器参数组
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

        return optimizer_grouped_parameters

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.hparams.model_name_or_path,
            "language": self.language,
            "is_chinese_model": self.is_chinese_model,
            "output_size": self.hparams.output_size,
            "hidden_size": self.config.hidden_size,
            "learning_rate": self.hparams.learning_rate,
            "batch_size": self.hparams.train_batch_size,
            "max_epochs": self.trainer.max_epochs if hasattr(self, 'trainer') else None,
        }