from pytorch_lightning.callbacks import Callback
from src.modules.runner import EvalRunner,SaveRunner,NoLabelRunner
import json
from torchmetrics import MetricCollection

from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

class ModelEvaluationCallback(Callback):
    """
    This callback is used to compute labelled metrics.
    """
    def __init__(self,explanator,dataset_eval,save_file: str,metrics,run_every_x : int=1,reshape_transform=None):
        """
        Initialization of ModelEvaluationCallback
        Args:
            explanator (CAM): pytorch_grad_cam method to compute activation map.
            dataset_eval (Dataset): evaluation dataset.
            save_file (str): path to file.
            metrics (list of Metrics or MetricCollection): collection of metrics to run.
            run_every_x (int, optional): How often your callback will be runned. Defaults to 1.
        """

        self.explanator = explanator
        
        self.dataset_eval = dataset_eval
        self.save_file = save_file
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        
        self.results = {metric_name.__class__.__name__ : [] for metric_name in metrics}
        self.run_every_x=run_every_x
        self.epoch = 0
        self.reshape_transform=reshape_transform
    
    def on_train_epoch_start(self, trainer, pl_module):
        """
        This script will automatically run after each epoch.

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
            pl_module (pl.module): Pytoch lightning module
        """
        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return

        model = pl_module
        model.eval()
        target_layers = [model.last_conv_layer()]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=self.reshape_transform)
        cam.batch_size=40

     
        runner = EvalRunner(explanator=cam,dataset=self.dataset_eval,metrics=self.metrics,device=model.device)
        metrics_now = runner.run()
        for metric in metrics_now:
            self.results[metric].append(metrics_now[metric].item()/runner.length)
        runner.save_metrics(self.results,to_save=self.save_file)
        model.train()
        self.metrics.reset()



class ModelImageSaveCallback(Callback):
    """
    This callback is used to save activation map and compute image-vise metrics for each image.
    """
    def __init__(self,explanator,dataset_eval,save_directory,metrics,run_every_x=1,reshape_transform=None):
        """
        Initialization of ModelImageSaveCallback
        Args:
            explanator (CAM): pytorch_grad_cam method to compute activation map.
            dataset_eval (Dataset): evaluation dataset.
            save_file (str): path to file.
            metrics (list of Metrics or MetricCollection): collection of metrics to run.
            run_every_x (int, optional): How often your callback will be runned. Defaults to 1.
        """
        self.explanator = explanator
        self.dataset_eval = dataset_eval
        self.save_directory = save_directory
        self.run_every_x = run_every_x
        self.epoch = 0
        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics = MetricCollection(*metrics)
        elif isinstance(metrics, MetricCollection):
            self.metrics = metrics
        self.reshape_transform = reshape_transform
        
    def on_train_epoch_start(self, trainer, pl_module):
        """
        This script will automatically run after each epoch.

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
            pl_module (pl.module): Pytoch lightning module
        """
        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return

        model = pl_module
        model.eval()
        target_layers = [model.last_conv_layer()]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=self.reshape_transform)
        cam.batch_size=40


        runner = SaveRunner(explanator=cam,dataset=self.dataset_eval,metrics=self.metrics)
        runner.run(model,self.epoch,self.save_directory)
        model.train()
        self.metrics.reset()







class NoLabelCallback(Callback):
    """
    This callback is used to compute no label metrics.
    """
    def __init__(self,explanator,dataset_eval,save_file,metrics,run_every_x=10,reshape_transform=None):
        """
        Initialization of NoLabelCallback
        Args:
            explanator (CAM): pytorch_grad_cam method to compute activation map.
            dataset_eval (Dataset): evaluation dataset.
            save_file (str): path to file.
            metrics (list of Metrics or MetricCollection): collection of metrics to run.
            run_every_x (int, optional): How often your callback will be runned. Defaults to 1.
        """
        self.explanator = explanator
        self.dataset_eval = dataset_eval
        self.save_file = save_file
        self.results = {i:[] for i in dataset_eval.CLASS_LABELS_LIST}
        self.start = True
        self.run_every_x = run_every_x
        self.epoch = 0
        self.cam_metric = metrics
        self.reshape_transform = reshape_transform
    def on_train_epoch_start(self, trainer, pl_module):
        """
        This script will automatically run after each epoch.

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
            pl_module (pl.module): Pytoch lightning module
        """
        self.epoch+=1
        if self.run_every_x != 1 and self.epoch % self.run_every_x !=1:
            return
        model = pl_module
        model.eval()
        target_layers = [model.last_conv_layer()]

        cam = self.explanator(model=model, target_layers=target_layers, use_cuda=True,reshape_transform=self.reshape_transform)
        cam.batch_size=40

        runner = NoLabelRunner(explanator=cam,dataset=self.dataset_eval,metrics= self.cam_metric)
        scores = runner.run(model)
        for i in scores:
            for j in range(len(scores[i])):
                if self.start:
                    self.results[i].append([scores[i][j]])
                else:
                    self.results[i][j].append(scores[i][j])
        self.start=False
        with open(self.save_file, 'w') as fp:
            json.dump(self.results, fp)
    
        model.train()

