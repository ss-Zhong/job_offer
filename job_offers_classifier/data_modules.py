from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Optional

class TransformerDataModule(LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        model_name_or_path: str,
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        shuffle_train: bool = True,
        verbose: bool = True,
        # 新增中文tokenizer配置
        tokenizer_config: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.train_shuffle = shuffle_train
        self.tokenizer = None
        self.verbose = verbose
        
        # 中文tokenizer配置优化
        self.tokenizer_config = tokenizer_config or {}
        
        # 为中文模型自动优化tokenizer配置
        if self._is_chinese_model(model_name_or_path):
            default_chinese_config = {
                'do_lower_case': True,
                'tokenize_chinese_chars': True,
                'strip_accents': False,
                'use_fast': True
            }
            default_chinese_config.update(self.tokenizer_config)
            self.tokenizer_config = default_chinese_config
            
            if self.verbose:
                print(f"✓ 检测到中文模型，应用中文tokenizer优化")

        if self.verbose:
            print(f"Initializing TransformerDataModule with model_name={model_name_or_path}, max_seq_length={max_seq_length}, train/eval_batch_size={train_batch_size}/{eval_batch_size}, num_workers={num_workers} ...")

    def _is_chinese_model(self, model_name_or_path: str) -> bool:
        """检测是否为中文模型"""
        chinese_indicators = ['chinese', 'hfl', 'ckiplab', 'bert-base-chinese', 'macbert', 'roberta-wwm']
        return any(indicator in model_name_or_path.lower() for indicator in chinese_indicators)

    def _get_dataloader(self, dataset_key, batch_size=32, shuffle=False):
        if dataset_key in self.dataset:
            return DataLoader(
                self.dataset[dataset_key],
                batch_size=batch_size,
                num_workers=self.num_workers,
                shuffle=shuffle,
                # Windows兼容性配置
                persistent_workers=False,  # 关闭持久worker避免Windows问题
                pin_memory=True if self.num_workers == 0 else False  # 只在单进程时使用pin_memory
            )
        else:
            return None

    def setup(self, stage=None):
        if self.verbose:
            print("Setting up TransformerDataModule ...")

        # 使用优化配置加载tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, 
                **self.tokenizer_config
            )
            
            if self.verbose:
                print(f"✓ Successfully loaded tokenizer for {self.model_name_or_path}")
                if hasattr(self.tokenizer, 'vocab_size'):
                    print(f"  Vocab size: {self.tokenizer.vocab_size}")
                    
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            # 回退到基础配置
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                use_fast=True
            )
            print(f"✓ Fallback tokenizer loaded")
        
        for subset in self.dataset.values():
            subset.setup(self.tokenizer, self.max_seq_length)

    def train_dataloader(self):
        return self._get_dataloader("train", batch_size=self.train_batch_size, shuffle=self.train_shuffle)

    def val_dataloader(self):
        return self._get_dataloader("val", batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return self._get_dataloader("test", batch_size=self.train_batch_size)