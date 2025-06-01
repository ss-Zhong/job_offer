#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文职业分类器模块
支持层次化损失和多任务学习
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.datasets import dump_svmlight_file
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import jieba
import jieba.posseg as pseg
from zhon.hanzi import punctuation as chinese_punctuation
from napkinxc.models import HSM, OVR

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# 基础导入
try:
    from job_offers_classifier.load_save import save_obj, load_obj
except ImportError:
    # 如果找不到，使用简单的pickle实现
    import pickle
    def save_obj(path, obj):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    
    def load_obj(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# 检查增强版模块
try:
    from enhanced_transformer_module import EnhancedTransformerClassifier
    from hierarchical_loss import HierarchicalISCOLoss, HierarchicalMultitaskLoss
    ENHANCED_AVAILABLE = True
    print("✅ 增强版TransformerClassifier可用")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    print(f"⚠️ 增强版TransformerClassifier不可用: {e}")


# 简单的数据集类
class TextDataset(Dataset):
    """文本数据集"""
    def __init__(self, texts, labels=None, num_labels=None, 
                 lazy_encode=True, labels_dense_vec=False, labels_groups=None):
        self.texts = texts
        self.labels = labels
        self.num_labels = num_labels
        self.lazy_encode = lazy_encode
        self.labels_dense_vec = labels_dense_vec
        self.labels_groups = labels_groups
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        item = {'text': self.texts[idx]}
        if self.labels is not None:
            if self.labels_dense_vec:
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            else:
                item['labels'] = self.labels[idx]
        return item


# 简单的Transformer分类器
class TransformerClassifier(pl.LightningModule):
    """基础Transformer分类器"""
    
    def __init__(self,
                 model_name_or_path,
                 output_size,
                 learning_rate=2e-5,
                 adam_epsilon=1e-8,
                 weight_decay=0.01,
                 train_batch_size=16,
                 eval_batch_size=16,
                 verbose=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        # 加载预训练模型
        self.transformer = AutoModel.from_pretrained(model_name_or_path)
        self.dropout = nn.Dropout(0.1)
        
        # 输出层
        hidden_size = self.transformer.config.hidden_size
        self.output = nn.Linear(hidden_size, output_size)
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        if verbose:
            print(f"✅ TransformerClassifier 初始化完成")
            print(f"   模型: {model_name_or_path}")
            print(f"   输出大小: {output_size}")
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.output(pooled_output)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)
        labels = batch['labels']
        
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)
        labels = batch['labels']
        
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.criterion(logits, labels)
        
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch.get('token_type_ids', None)
        
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probs = torch.softmax(logits, dim=-1)
        
        return probs
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
    
    def save_transformer(self, save_path):
        """保存transformer模型"""
        self.transformer.save_pretrained(save_path)


# 简化的DataModule
class TransformerDataModule(pl.LightningDataModule):
    """Transformer数据模块"""
    
    def __init__(self, 
                 datasets,
                 tokenizer_name,
                 train_batch_size=16,
                 eval_batch_size=16,
                 max_seq_length=512,
                 num_workers=0,
                 tokenizer_config=None):
        super().__init__()
        self.datasets = datasets
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
    
    def setup(self, stage=None):
        # 这里可以进行数据预处理
        pass
    
    def collate_fn(self, batch):
        """批处理函数"""
        texts = [item['text'] for item in batch]
        
        # 编码文本
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        # 添加标签
        if 'labels' in batch[0]:
            labels = torch.tensor([item['labels'] for item in batch])
            encoded['labels'] = labels
        
        return encoded
    
    def train_dataloader(self):
        if 'train' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['train'],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        if 'val' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['val'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        if 'test' not in self.datasets:
            return None
        return DataLoader(
            self.datasets['test'],
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )


# 简化的Trainer包装器
class TrainerWrapper:
    """PyTorch Lightning Trainer的简单包装器"""
    
    def __init__(self, ckpt_dir=None, trainer_args=None, early_stopping=False, 
                 early_stopping_args=None, verbose=True):
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose
        
        # 默认训练器参数
        default_args = {
            'max_epochs': 10,
            'devices': 1,
            'accelerator': 'auto',
            'precision': 16,
            'enable_progress_bar': verbose,
            'enable_model_summary': verbose
        }
        
        if trainer_args:
            default_args.update(trainer_args)
        
        # 回调
        callbacks = []
        if early_stopping:
            from pytorch_lightning.callbacks import EarlyStopping
            es_args = early_stopping_args or {}
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=es_args.get('patience', 3),
                min_delta=es_args.get('min_delta', 0.001),
                mode='min'
            ))
        
        if ckpt_dir:
            from pytorch_lightning.callbacks import ModelCheckpoint
            callbacks.append(ModelCheckpoint(
                dirpath=ckpt_dir,
                filename='model-{epoch:02d}-{val_loss:.4f}',
                save_top_k=1,
                monitor='val_loss',
                mode='min'
            ))
        
        default_args['callbacks'] = callbacks
        
        self.trainer = pl.Trainer(**default_args)
    
    def fit(self, model, datamodule, ckpt_path=None):
        """训练模型"""
        self.trainer.fit(model, datamodule, ckpt_path=ckpt_path)
    
    def predict(self, model, dataloaders, ckpt_path=None):
        """预测"""
        return self.trainer.predict(model, dataloaders, ckpt_path=ckpt_path)


class BaseHierarchicalJobOffersClassifier:
    """基础层次化职业分类器"""
    
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 modeling_mode='bottom-up',
                 verbose=True):
        self.model_dir = model_dir
        self.hierarchy = hierarchy
        self.modeling_mode = modeling_mode
        self.base_model = None
        self.verbose = verbose

    def _init_fit(self):
        if self.model_dir is None:
            raise RuntimeError("Cannot fit with model_dir = None")

        if self.hierarchy is None:
            raise RuntimeError("Cannot fit with hierarchy = None")

        os.makedirs(self.model_dir, exist_ok=True)
        self.hierarchy_path = os.path.join(self.model_dir, "hierarchy.bin")
        save_obj(self.hierarchy_path, self.hierarchy)
        self._process_hierarchy()

    def _process_y(self, y):
        return [self.last_level_labels_map[y_i] for y_i in y]

    def _get_level_labels(self, level):
        level_labels = sorted([node['label'] for node in self.hierarchy.values() if node['level'] == level])
        return level_labels

    def _process_hierarchy(self):
        # Basic per level information
        self.levels = sorted({node['level'] for node in self.hierarchy.values()})
        self.levels_labels = {level: self._get_level_labels(level) for level in self.levels}
        self.levels_labels_map = {level: {label: i for i, label in enumerate(sorted(labels))} for level, labels in self.levels_labels.items()}
        self.levels_indices_map = {level: {i: label for label, i in labels_map.items()} for level, labels_map in self.levels_labels_map.items()}
        self.level_labels_count = {level: len(labels) for level, labels in self.levels_labels.items()}

        # Last level information
        self.last_level = max(self.levels)
        self.last_level_labels = self.levels_labels[self.last_level]
        self.last_level_labels_map = self.levels_labels_map[self.last_level]
        self.last_level_indices_map = self.levels_indices_map[self.last_level]
        self.last_level_labels_count = self.level_labels_count[self.last_level]

    def _init_load(self, model_dir):
        self.model_dir = model_dir
        self.hierarchy_path = os.path.join(self.model_dir, "hierarchy.bin")
        self.hierarchy = load_obj(self.hierarchy_path)
        self._process_hierarchy()

    def _get_output(self, pred, output_level="last", format="array", top_k=None, pred_type="flat"):
        if output_level == "last":
            output_level = self.last_level
        
        level_pred = pred
        level_map = self.last_level_indices_map

        # Get top_k labels
        if top_k is not None:
            if not isinstance(top_k, int) or top_k < 1:
                raise ValueError(f"top_k needs to be int > 0, is {top_k}")

            top_k_labels = np.flip(np.argsort(level_pred, axis=1), axis=1)[:, :top_k]
            top_k_prob = np.take_along_axis(level_pred, top_k_labels, axis=1)
            level_pred = (top_k_labels, top_k_prob)

        # Apply requested format
        if format == "array":
            return level_pred, level_map
        elif format == "dataframe":
            if top_k is None:
                columns = [v for i, v in sorted(list(level_map.items()))]
                return pd.DataFrame(level_pred, columns=columns)
            else:
                columns = [f"class_{i + 1}" for i in range(top_k)] + [f"prob_{i + 1}" for i in range(top_k)]
                df = pd.DataFrame(np.hstack(level_pred), columns=columns)
                for c in columns[:top_k]:
                    df[c] = df[c].apply(lambda x: level_map[int(x)])
                return df
        else:
            raise ValueError(f"Unknown format {format}")


class ChineseTransformerJobOffersClassifier(BaseHierarchicalJobOffersClassifier):
    """中文BERT职业分类器 - 支持层次化损失"""
    
    def __init__(self,
                 model_dir=None,
                 hierarchy=None,
                 ckpt_path=None,
                 transformer_model="hfl/chinese-roberta-wwm-ext",
                 transformer_ckpt_path="",
                 modeling_mode='bottom-up',
                 adam_epsilon=1e-8,
                 learning_rate=2e-5,
                 weight_decay=0.01,
                 max_epochs=20,
                 batch_size=16,
                 max_sequence_length=256,
                 early_stopping=True,
                 early_stopping_delta=0.001,
                 early_stopping_patience=3,
                 devices=1,
                 accelerator="auto",
                 num_nodes=1,
                 threads=-1,
                 precision=16,
                 verbose=True,
                 # 层次化参数
                 use_hierarchical_loss=False,
                 use_multitask_learning=False,
                 hierarchical_loss_weights=None,
                 task_weights=None):
        
        super().__init__(model_dir, hierarchy, modeling_mode, verbose)
        
        # 保存所有参数
        self.ckpt_path = ckpt_path
        self.transformer_model = transformer_model
        self.transformer_ckpt_path = transformer_ckpt_path
        self.adam_epsilon = adam_epsilon
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.early_stopping = early_stopping
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.devices = devices
        self.accelerator = accelerator
        self.num_nodes = num_nodes
        self.threads = threads if threads != -1 else os.cpu_count()
        self.precision = precision
        
        # 层次化参数
        self.use_hierarchical_loss = use_hierarchical_loss and ENHANCED_AVAILABLE
        self.use_multitask_learning = use_multitask_learning and ENHANCED_AVAILABLE
        self.hierarchical_loss_weights = hierarchical_loss_weights or {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0}
        self.task_weights = task_weights or {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
        
        if self.verbose:
            print(f"✅ 中文BERT分类器初始化完成")
            print(f"   模型: {transformer_model}")
            print(f"   层次化损失: {'启用' if self.use_hierarchical_loss else '禁用'}")
            print(f"   多任务学习: {'启用' if self.use_multitask_learning else '禁用'}")

    def _create_text_dataset(self, y, X, labels_groups=None):
        """创建文本数据集"""
        return TextDataset(
            X, 
            labels=y,
            num_labels=self.last_level_labels_count if hasattr(self, 'last_level_labels_count') else None,
            lazy_encode=True, 
            labels_dense_vec=False, 
            labels_groups=labels_groups
        )

    def _setup_data_module(self, dataset):
        """设置数据模块"""
        text_dataset = {}
        
        if 'train' in dataset:
            text_dataset['train'] = self._create_text_dataset(
                *dataset['train'], 
                labels_groups=dataset.get('labels_groups', None)
            )
        
        if 'val' in dataset:
            text_dataset['val'] = self._create_text_dataset(
                *dataset['val'], 
                labels_groups=dataset.get('labels_groups', None)
            )
        
        if 'test' in dataset:
            text_dataset['test'] = self._create_text_dataset(
                *dataset['test'], 
                labels_groups=dataset.get('labels_groups', None)
            )
        
        data_module = TransformerDataModule(
            text_dataset,
            self.transformer_model,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            max_seq_length=self.max_sequence_length,
            num_workers=self.threads if self.devices == 1 else 0
        )
        
        data_module.setup()
        return data_module

    def _fit(self, y, X, y_val=None, X_val=None, output_size=None, 
             labels_groups=None, labels_paths=None, labels_groups_mapping=None):
        """内部训练方法"""
        if output_size is None:
            raise RuntimeError("Output size is not provided")

        dataset = {"train": (y, X)}

        if y_val is not None and X_val is not None:
            dataset["val"] = (y_val, X_val)
        
        if labels_groups is not None:
            dataset['labels_groups'] = labels_groups

        data_module = self._setup_data_module(dataset)
        
        trainer = TrainerWrapper(
            ckpt_dir=os.path.join(self.model_dir, "ckpts"),
            trainer_args={
                "max_epochs": self.max_epochs,
                "devices": self.devices,
                "num_nodes": self.num_nodes,
                "precision": self.precision,
                "accelerator": self.accelerator
            },
            early_stopping=self.early_stopping,
            early_stopping_args={
                "patience": self.early_stopping_patience,
                "min_delta": self.early_stopping_delta
            },
            verbose=self.verbose
        )

        # 创建模型
        if (self.use_hierarchical_loss or self.use_multitask_learning) and ENHANCED_AVAILABLE:
            # 使用增强版
            if self.verbose:
                print("🔧 使用增强版TransformerClassifier")
            
            self.base_model = EnhancedTransformerClassifier(
                model_name_or_path=self.transformer_model,
                output_size=output_size,
                adam_epsilon=self.adam_epsilon,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                verbose=self.verbose,

                isco_hierarchy=self.hierarchy,
                use_hierarchical_loss=self.use_hierarchical_loss,
                use_multitask_learning=self.use_multitask_learning,
                hierarchical_loss_weights=self.hierarchical_loss_weights,
                task_weights=self.task_weights,
            )

        else:
            # 使用标准版
            if self.verbose:
                print("📝 使用标准版TransformerClassifier")
            
            self.base_model = TransformerClassifier(
                model_name_or_path=self.transformer_model,
                output_size=output_size,
                adam_epsilon=self.adam_epsilon,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                verbose=self.verbose,
            )

        trainer.fit(self.base_model, datamodule=data_module, ckpt_path=self.ckpt_path)
        self.base_model.save_transformer(self.model_dir)

    def fit(self, y, X, y_val=None, X_val=None):
        """训练模型"""
        self._init_fit()

        # 保存架构信息
        arch_info = {
            'transformer_model': self.transformer_model,
            'transformer_ckpt': self.transformer_ckpt_path,
            'use_hierarchical_loss': self.use_hierarchical_loss,
            'use_multitask_learning': self.use_multitask_learning,
            'hierarchical_loss_weights': self.hierarchical_loss_weights,
            'task_weights': self.task_weights,
            'enhanced_available': ENHANCED_AVAILABLE
        }
        
        save_obj(os.path.join(self.model_dir, "transformer_arch.bin"), arch_info)

        if self.verbose:
            print(f"🎯 开始训练，样本数: {len(y)}")
            if y_val is not None:
                print(f"   验证集: {len(y_val)} 样本")

        # 处理标签
        y = self._process_y(y)
        if y_val is not None:
            y_val = self._process_y(y_val)

        # 训练
        self._fit(y, X, y_val=y_val, X_val=X_val, 
                 output_size=self.last_level_labels_count)

    def load(self, model_dir):
        """加载模型"""
        self._init_load(model_dir)

        # 读取架构信息
        arch_info = load_obj(os.path.join(self.model_dir, "transformer_arch.bin"))
        self.transformer_model = arch_info['transformer_model']
        self.transformer_ckpt_path = arch_info.get('transformer_ckpt', '')
        
        # 读取层次化配置
        self.use_hierarchical_loss = arch_info.get('use_hierarchical_loss', False)
        self.use_multitask_learning = arch_info.get('use_multitask_learning', False)
        self.hierarchical_loss_weights = arch_info.get('hierarchical_loss_weights', {1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0})
        self.task_weights = arch_info.get('task_weights', {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4})

        # 创建模型
        if (self.use_hierarchical_loss or self.use_multitask_learning) and ENHANCED_AVAILABLE:
            self.base_model = EnhancedTransformerClassifier(
                model_name_or_path=self.model_dir,
                output_size=self.last_level_labels_count,
                isco_hierarchy=self.hierarchy,
                use_hierarchical_loss=self.use_hierarchical_loss,
                use_multitask_learning=self.use_multitask_learning,
                hierarchical_loss_weights=self.hierarchical_loss_weights,
                task_weights=self.task_weights,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                verbose=self.verbose,
            )
        else:
            self.base_model = TransformerClassifier(
                model_name_or_path=self.model_dir,
                output_size=self.last_level_labels_count,
                train_batch_size=self.batch_size,
                eval_batch_size=self.batch_size,
                verbose=self.verbose,
            )

        # 查找checkpoint
        self.ckpt_path = self._find_checkpoint()

    def _find_checkpoint(self):
        """查找checkpoint文件"""
        ckpt_path = os.path.join(self.model_dir, "transformer_classifier.ckpt")
        
        if not os.path.exists(ckpt_path):
            ckpts_dir = os.path.join(self.model_dir, "ckpts")
            if os.path.exists(ckpts_dir):
                ckpt_files = [f for f in os.listdir(ckpts_dir) if f.endswith(".ckpt")]
                if ckpt_files:
                    # 按时间排序，取最新的
                    ckpt_files.sort()
                    ckpt_path = os.path.join(ckpts_dir, ckpt_files[-1])
        
        if os.path.exists(ckpt_path):
            if self.verbose:
                print(f"✓ 找到checkpoint: {ckpt_path}")
            return ckpt_path
        else:
            raise RuntimeError(f"找不到checkpoint在 {self.model_dir}")

    def _predict(self, X):
        """内部预测方法"""
        if self.base_model is None:
            raise RuntimeError("模型未训练或加载")

        dataset = {"test": (None, X)}
        data_module = self._setup_data_module(dataset)

        trainer = TrainerWrapper(
            ckpt_dir=self.ckpt_path,
            trainer_args={"devices": self.devices, "precision": self.precision}
        )
        
        pred = trainer.predict(self.base_model, 
                              dataloaders=data_module.test_dataloader(), 
                              ckpt_path=self.ckpt_path)
        
        return np.array(torch.vstack(pred))

    def predict(self, X, output_level="last", format='array', top_k=None):
        """预测"""
        if self.verbose:
            print(f"🔮 预测 {len(X)} 个样本...")
        
        pred = self._predict(X)
        return self._get_output(pred, output_level=output_level, 
                               format=format, top_k=top_k)


# 工具函数
def get_recommended_chinese_models():
    """获取推荐的中文模型"""
    return {
        'roberta': {
            'model_name': 'hfl/chinese-roberta-wwm-ext',
            'description': 'HFL Chinese RoBERTa (推荐) - 性能最好',
            'recommended': True
        },
        'bert': {
            'model_name': 'hfl/chinese-bert-wwm-ext',
            'description': 'HFL Chinese BERT - 经典稳定',
            'recommended': True
        },
        'macbert': {
            'model_name': 'hfl/chinese-macbert-base',
            'description': 'HFL Chinese MacBERT - 改进架构',
            'recommended': True
        },
        'google': {
            'model_name': 'bert-base-chinese',
            'description': 'Google Chinese BERT - 基础模型',
            'recommended': False
        }
    }


if __name__ == "__main__":
    print("🎯 中文职业分类器模块")
    print("=" * 50)
    
    print("\n推荐的中文BERT模型:")
    for key, info in get_recommended_chinese_models().items():
        status = "⭐" if info['recommended'] else "  "
        print(f"  {status} {key}: {info['model_name']} - {info['description']}")
    
    print("\n使用示例:")
    print("""
    # 创建层级结构
    from job_offers_utils import create_hierarchy_from_isco_codes
    hierarchy = create_hierarchy_from_isco_codes(['1121', '1122', '2121'])
    
    # 标准模式（无层次化损失）
    classifier = ChineseTransformerJobOffersClassifier(
        model_dir='./models/standard',
        hierarchy=hierarchy,
        transformer_model='hfl/chinese-roberta-wwm-ext',
        max_epochs=5
    )
    
    # 层次化损失模式
    classifier_hierarchical = ChineseTransformerJobOffersClassifier(
        model_dir='./models/hierarchical',
        hierarchy=hierarchy,
        transformer_model='hfl/chinese-roberta-wwm-ext',
        use_hierarchical_loss=True,
        hierarchical_loss_weights={1: 8.0, 2: 4.0, 3: 2.0, 4: 1.0},
        max_epochs=5
    )
    
    # 训练模型
    texts = ["财务经理职责...", "软件工程师..."]
    labels = ["1121", "2512"]
    classifier.fit(labels, texts)
    
    # 预测
    predictions = classifier.predict(new_texts, format='dataframe', top_k=5)
    """)