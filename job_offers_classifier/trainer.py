from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TrainerWrapper:
    def __init__(self,
                 monitor="val_loss",
                 ckpt_dir=None,
                 trainer_args=None,
                 early_stopping=False,
                 early_stopping_args=None,
                 checkpoint_args=None,
                 verbose=True):

        self.monitor = monitor
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose

        self.early_stopping_args = {
            "monitor": monitor,
            "min_delta": 0.001,
            "patience": 5,
            "mode": "min",
            "verbose": verbose,
        }
        if early_stopping_args:
            self.early_stopping_args.update(early_stopping_args)

        self.checkpoint_args = {
            "filename": "{epoch}-{" + monitor + ":.5f}",
            "monitor": monitor,
            "mode": "min",
            "save_top_k": 3,
            "verbose": verbose,
        }
        if checkpoint_args:
            self.checkpoint_args.update(checkpoint_args)

        callbacks = [ModelCheckpoint(self.ckpt_dir, **self.checkpoint_args)]
        if early_stopping:
            callbacks.append(EarlyStopping(**self.early_stopping_args))

        self.trainer_args = {
            #"gpus": 1,
            "devices": 1,
            #"amp_backend": "native", # Deprecated
            "accelerator": "auto",
            #"auto_select_gpus": True, # Deprecated
            "precision": 32,
            "callbacks": callbacks,
            "enable_model_summary": verbose,
            "enable_progress_bar": verbose,
            "num_sanity_val_steps": 0,
        }
        if trainer_args:
            self.trainer_args.update(trainer_args)

        self.trainer = Trainer(**self.trainer_args)

    def fit(self, module, **kwargs):
        if self.verbose:
            print(f"Starting training {module.__class__.__name__} ...")

        self.trainer.fit(module, **kwargs)

    def test(self, module, **kwargs):
        if self.verbose:
            print(f"Starting testing {module.__class__.__name__} ...")

        self.trainer.test(module, **kwargs)

    def predict(self, module, **kwargs):
        if self.verbose:
            print(f"Starting predicting with {module.__class__.__name__} ...")

        return self.trainer.predict(module, **kwargs)

    def save_checkpoint(self, chkp_path):
        self.trainer.save_checkpoint(chkp_path)
