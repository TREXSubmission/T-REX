from config.scene_mc import SceneMC
from config.Pvoc_mc import PvocMC
from config.resnet_gender_mc import ResNetGenderMC

from src.datasets.gender import GenderClassificationAttentionDataset,GenderClassificationDataset
from src.datasets.Pvoc import PvocAttentionDataset,PvocClassificationDataset
from src.datasets.scene import SceneClassificationDataset,SceneClassificationAttentionDataset

def get_config(args):
    """
    Function to generate config file and dataset

    Args:
        args (argparser): argparser with config_file 

    Returns:
        config, dataset - python config class, Dataset for training
    """
    if args.config_file == 'gender':
        config = ResNetGenderMC()
        return config,GenderClassificationDataset(
            root_path=config.root_path, split='test', resize_size=config.eval_resize_size)

        
    elif args.config_file == 'pvoc':
        config = PvocMC()

        return config,PvocClassificationDataset(
            root_path='/home/bovey/datasets/data/ML-Interpretability-Evaluation-Benchmark',resize_size=config.eval_resize_size)
    
    
    elif args.config_file == 'scene':
        config = SceneMC()
        return config,SceneClassificationDataset(
                root_path=config.root_path, split='test', resize_size=config.eval_resize_size)




def get_config_original(args):
    """
    Function to generate config file and eval dataset

    Args:
        args (argparser): argparser with config_file 

    Returns:
        config, dataset - python config class, Dataset for evaluation
    """
    if args.config_file == 'gender':
        config = ResNetGenderMC()
        return config
        
    elif args.config_file == 'pvoc':
        config = PvocMC()

        return config
    
    elif args.config_file == 'scene':
        config = SceneMC()
        return config