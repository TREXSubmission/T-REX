"""adapted from https://github.com/microsoft/robust-models-transfer"""
import argparse
import os
import cox.store
import torch as ch
from robustness import datasets, defaults, model_utils, train
from robustness.tools import helpers
from torchvision import models

import torch
from torchmetrics import Accuracy, MetricCollection
from src.datasets import Pvoc
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np

import torch.nn
from torch import nn
from robustness.tools.custom_modules import SequentialWithArgs
from config.adv_resnet_pvoc_mc import AdvPvocMC

config = AdvPvocMC()

parser = argparse.ArgumentParser(description='Transfer learning via pretrained Imagenet models',
                                 conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)


# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar',
                    help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--resume', action='store_true',
                    help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--pytorch-pretrained', action='store_true',
                    help='If True, loads a Pytorch pretrained model.')
parser.add_argument('--cifar10-cifar10', action='store_true',
                    help='cifar10 to cifar10 transfer')
parser.add_argument('--subset', type=int, default=None,
                    help='number of training data to use from the dataset')
parser.add_argument('--no-tqdm', type=int, default=1,
                    choices=[0, 1], help='Do not use tqdm.')
parser.add_argument('--no-replace-last-layer', action='store_true',
                    help='Whether to avoid replacing the last layer')
parser.add_argument('--freeze-level', type=int, default=-1,
                    help='Up to what layer to freeze in the pretrained model (assumes a resnet architectures)')
parser.add_argument('--additional-hidden', type=int, default=0,
                    help='How many hidden layers to add on top of pretrained network + classification layer')
parser.add_argument('--per-class-accuracy', action='store_true', help='Report the per-class accuracy. '
                                                                      'Can be used only with pets, caltech101,'
                                                                      ' caltech256, aircraft, and flowers.')


class TransformedDataset(Dataset):
    def __init__(self, ds, transform=None):
        self.transform = transform
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample, label = self.ds[idx]
        if self.transform:
            sample = self.transform(sample)
            if sample.shape[0] == 1:
                sample = sample.squeeze()
        return sample, label

def fine_tunify(model_name, model_ft, num_classes, additional_hidden=0):
    if model_name in ["resnet", "resnet18", "resnet50", "wide_resnet50_2", "wide_resnet50_4", "resnext50_32x4d", 'shufflenet']:
        num_ftrs = model_ft.fc.in_features
        # The two cases are split just to allow loading
        # models trained prior to adding the additional_hidden argument
        # without errors
        if additional_hidden == 0:
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
        else:
            model_ft.fc = SequentialWithArgs(
                *list(sum([[nn.Linear(num_ftrs, num_ftrs), nn.ReLU()] for i in range(additional_hidden)], [])),
                nn.Linear(num_ftrs, num_classes)
            )
        input_size = 224
    elif model_name == "alexnet":
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif "vgg" in model_name:
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
    elif model_name == "squeezenet":
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224
    elif model_name == "densenet":
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name in ["mnasnet", "mobilenet"]:
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    else:
        raise ValueError("Invalid model type, exiting...")

    return model_ft

def make_loaders_pvoc(batch_size, workers):
    root_path = '/home/mtemn/Documents/explainable-ai/data'
    train_resize_size = 224
    eval_resize_size = 224

    train_ds = Pvoc.PvocClassificationDataset(root_path, image_set='train', resize_size=train_resize_size)
    val_ds = Pvoc.PvocClassificationDataset(root_path, image_set='val', resize_size=eval_resize_size)
    train_ds = TransformedDataset(train_ds, transform=config.augmentation)
    # val_ds = TransformedDataset(val_ds, transform=cs.TEST_TRANSFORMS)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    return 20, (train_dl, val_dl)

DS_TO_FUNC = {
    "pvoc": make_loaders_pvoc,
}

PYTORCH_MODELS = {
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'mnasnet': models.mnasnet1_0,
}

def get_per_class_accuracy(args, loader):
    """Returns the custom per_class_accuracy function. When using this custom function
    look at only the validation accuracy. Ignore training set accuracy.
    """

    def _get_class_weights(args, loader):
        """Returns the distribution of classes in a given dataset.
        """
        if args.dataset in ['pets', 'flowers']:
            targets = loader.dataset.targets

        elif args.dataset in ['caltech101', 'caltech256']:
            targets = np.array([loader.dataset.ds.dataset.y[idx]
                                for idx in loader.dataset.ds.indices])

        elif args.dataset == 'aircraft':
            targets = [s[1] for s in loader.dataset.samples]

        counts = np.unique(targets, return_counts=True)[1]
        class_weights = counts.sum() / (counts * len(counts))
        return ch.Tensor(class_weights)

    class_weights = _get_class_weights(args, loader)


    def custom_acc(logits, labels):
        """Returns the top1 accuracy, weighted by the class distribution.
        This is important when evaluating an unbalanced dataset.
        """
        batch_size = labels.size(0)
        maxk = min(5, logits.shape[-1])
        prec1, _ = helpers.accuracy(
            logits, labels, topk=(1, maxk), exact=True)

        normal_prec1 = prec1.sum(0, keepdim=True).mul_(100 / batch_size)
        weighted_prec1 = prec1 * class_weights[labels.cpu()].cuda()
        weighted_prec1 = weighted_prec1.sum(
            0, keepdim=True).mul_(100 / batch_size)

        return weighted_prec1.item(), normal_prec1.item()

    return custom_acc


def get_dataset_and_loaders(args):
    """Given arguments, returns a datasets object and the train and validation loaders.
    """
    def make_loaders(ds, batch_size, workers, subset):
        if subset: raise Exception(f'Subset not supported for the {ds} dataset')
        return DS_TO_FUNC[ds](batch_size, workers)

    ds, (train_loader, validation_loader) = make_loaders(
        args.dataset, args.batch_size, 8, args.subset)
    if type(ds) == int:
        new_ds = datasets.CIFAR("/tmp")
        new_ds.num_classes = ds
        new_ds.mean = ch.tensor([0., 0., 0.])
        new_ds.std = ch.tensor([1., 1., 1.])
        ds = new_ds
    return ds, train_loader, validation_loader


def resume_finetuning_from_checkpoint(args, ds, finetuned_model_path, pytorch_models=None):
    """Given arguments, dataset object and a finetuned model_path, returns a model
    with loaded weights and returns the checkpoint necessary for resuming training.
    """
    print('[Resuming finetuning from a checkpoint...]')
    if args.dataset in list(DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
        model, _ = model_utils.make_and_restore_model(
            arch=pytorch_models[args.arch](
                args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
            dataset=datasets.ImageNet(''), add_custom_forward=args.arch in pytorch_models.keys())
        while hasattr(model, 'model'):
            model = model.model
        model = fine_tunify(
            args.arch, model, ds.num_classes, args.additional_hidden)
        model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=finetuned_model_path,
                                                               add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
    else:
        model, checkpoint = model_utils.make_and_restore_model(
            arch=args.arch, dataset=ds, resume_path=finetuned_model_path)
    return model, checkpoint


def get_model(args, ds, pytorch_models=None):
    """Given arguments and a dataset object, returns an ImageNet model (with appropriate last layer changes to
    fit the target dataset) and a checkpoint.The checkpoint is set to None if noe resuming training.
    """
    finetuned_model_path = os.path.join(
        args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    if args.resume and os.path.isfile(finetuned_model_path):
        model, checkpoint = resume_finetuning_from_checkpoint(
            args, ds, finetuned_model_path)
    else:
        if args.dataset in list(DS_TO_FUNC.keys()) and not args.cifar10_cifar10:
            model, _ = model_utils.make_and_restore_model(
                arch=pytorch_models[args.arch](
                    args.pytorch_pretrained) if args.arch in pytorch_models.keys() else args.arch,
                dataset=datasets.ImageNet(''), resume_path=args.model_path, pytorch_pretrained=args.pytorch_pretrained,
                add_custom_forward=args.arch in pytorch_models.keys())
            checkpoint = None
        else:
            model, _ = model_utils.make_and_restore_model(arch=args.arch, dataset=ds,
                                                          resume_path=args.model_path,
                                                          pytorch_pretrained=args.pytorch_pretrained)
            checkpoint = None

        if not args.no_replace_last_layer and not args.eval_only:
            print(f'[Replacing the last layer with {args.additional_hidden} '
                  f'hidden layers and 1 classification layer that fits the {args.dataset} dataset.]')
            while hasattr(model, 'model'):
                model = model.model
            model = fine_tunify(
                args.arch, model, ds.num_classes, args.additional_hidden)
            model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds,
                                                                   add_custom_forward=args.additional_hidden > 0 or args.arch in pytorch_models.keys())
        else:
            print('[NOT replacing the last layer]')
    return model, checkpoint


def freeze_model(model, freeze_level):
    """
    Freezes up to args.freeze_level layers of the model (assumes a resnet model)
    """
    # Freeze layers according to args.freeze-level
    update_params = None
    if freeze_level != -1:
        # assumes a resnet architecture
        assert len([name for name, _ in list(model.named_parameters())
                    if f"layer{freeze_level}" in name]), "unknown freeze level (only {1,2,3,4} for ResNets)"
        update_params = []
        freeze = True
        for name, param in model.named_parameters():
            print(name, param.size())

            if not freeze and f'layer{freeze_level}' not in name:
                print(f"[Appending the params of {name} to the update list]")
                update_params.append(param)
            else:
                param.requires_grad = False

            if freeze and f'layer{freeze_level}' in name:
                # if the freeze level is detected stop freezing onwards
                freeze = False
    return update_params


def args_preprocess(args):
    """
    Fill the args object with reasonable defaults, and also perform a sanity check to make sure no
    args are missing.
    """
    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0

    if args.pytorch_pretrained:
        assert not args.model_path, 'You can either specify pytorch_pretrained or model_path, not together.'


    ALL_DS = list(DS_TO_FUNC.keys()) + \
             ['imagenet', 'breeds_living_9', 'stylized_imagenet']
    assert args.dataset in ALL_DS

    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    return args


if __name__ == "__main__":
    args = parser.parse_args()

    # myargs
    args.arch = config.arch
    args.dataset = config.dataset
    args.data = config.data
    args.out_dir = config.out_dir
    args.exp_name = config.exp_name
    args.epochs = config.epochs
    args.lr = config.lr
    args.step_lr = config.step_lr
    args.batch_size = config.batch_size
    args.weight_decay = config.weight_decay
    args.adv_train = config.adv_train

    args.model_path = config.model_path
    args.freeze_level = config.freeze_level
    args.log_iters = config.log_iters
    args.momentum = config.momentum

    args.constraint = config.constraint
    args.eps = config.eps
    args.attack_lr = config.attack_lr
    args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args.attack_steps = config.attack_steps
    # args.use_best =
    # args.random_restarts =

    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    # # Create store and log the args
    store = cox.store.Store(args.out_dir, args.exp_name)
    ds, train_loader, validation_loader = get_dataset_and_loaders(args)

    if args.per_class_accuracy:
        assert args.dataset in ['pets', 'caltech101', 'caltech256', 'flowers', 'aircraft'], \
            f'Per-class accuracy not supported for the {args.dataset} dataset.'

        # VERY IMPORTANT
        # We report the per-class accuracy using the validation
        # set distribution. So ignore the training accuracy (as you will see it go
        # beyond 100. Don't freak out, it doesn't really capture anything),
        # just look at the validation accuarcy
        args.custom_accuracy = get_per_class_accuracy(args, validation_loader)

    def custom_train_loss(logits, targ):
        return nn.BCEWithLogitsLoss()(logits, targ.to(torch.float32))

    def custom_adv_loss(logits, targ):
        return nn.BCEWithLogitsLoss(reduction='none')(logits, targ.to(torch.float32))

    def adv_loss(model, inp, targ):
        logits = model(inp)
        adv_out = custom_adv_loss(logits, targ).mean(dim=1)
        return adv_out, logits

    accuracy = Accuracy(average='macro', num_classes=20)
    def custom_acc(logits, targ):
        acc = accuracy(logits.to(torch.float32).to(ch.device('cpu:0')),
                       targ.to(torch.int32).to(ch.device('cpu:0')))
        return acc, acc

    args.custom_train_loss = custom_train_loss
    args.custom_adv_loss = adv_loss
    args.custom_accuracy = custom_acc
    model, checkpoint = get_model(args, ds, pytorch_models=PYTORCH_MODELS)
    if args.eval_only:
        train.eval_model(args, model, validation_loader, store=store)
    update_params = freeze_model(model, freeze_level=args.freeze_level)

    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    train.train_model(args, model, (train_loader, validation_loader), store=store,
                      checkpoint=checkpoint)
