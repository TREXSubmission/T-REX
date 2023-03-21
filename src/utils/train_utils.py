import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger
import pytorch_lightning as pl


class EquivariantModelCheckpoint(ModelCheckpoint):
    """
    modify checkpointing for equivariant escnn models so that the model
    is exported before making a valid checkpoint
    """
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        """
        Save checkpoint

        Args:
            trainer (pl.Trainer): pytorch lightning trainer
            filepath (str): path to file
        """
        trainer.save_checkpoint(filepath, self.save_weights_only)
        torch.save({
            'model_state_dict': trainer.model.model.export().state_dict(),
        }, filepath)
        trainer.model.model.train()
        self._last_global_step_saved = trainer.global_step
        from weakref import proxy

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))


def get_training_callbacks(config, log_path, experiment_name, equivariant=False):
    """
    Get callback for training from config file.

    Args:
        config (_type_): Config file
        log_path (_type_): path to log
        experiment_name (_type_): name of the experiment
        equivariant (bool, optional): Is your model is equivariant. Defaults to False.

    Returns:
        Callbacks: Callbacks, but only for training
    """
    ckpt_callback = ModelCheckpoint
    if equivariant:
        ckpt_callback = EquivariantModelCheckpoint
    checkpoint_callback = ckpt_callback(
        dirpath=log_path,
        filename='val_min_checkpoint_{epoch:02d}-{acc:02.0f}',
        mode='min',
        every_n_epochs=1,
        save_last=True,
        save_top_k=3,
        monitor='val_loss'
    )

    lr_monitor = LearningRateMonitor()
    callbacks = config.callbacks
    callbacks.append(checkpoint_callback)
    callbacks.append(lr_monitor)
    return callbacks


def get_training_loggers(config, log_path, experiment_name):
    """
    Get loggers for training from config file.

    Args:
        config (_type_): Config file
        log_path (_type_): path to log
        experiment_name (_type_): name of the experiment

    Returns:
        list of loggers
    """
    tb_logger = TensorBoardLogger(log_path, config.task, version=experiment_name)
    csv_logger = CSVLogger(log_path, config.task, version=experiment_name)
   
    return [tb_logger,csv_logger]
