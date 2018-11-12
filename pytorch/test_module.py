"""
A code to test the module of mean teacher
"""
from main import create_data_loaders
import mean_teacher.datasets  as datasets
from experiments.cifar10_test import parameters
from mean_teacher.cli import parse_dict_args
import torch

# print(next(parameters()))
param = next(parameters())

def parse_parameters(title, base_batch_size, base_labeled_batch_size, base_lr, n_labels, data_seed, **kwargs):
    ngpu = torch.cuda.device_count()
    adapted_args = {
            'batch_size': base_batch_size * ngpu,
            'labeled_batch_size': base_labeled_batch_size * ngpu,
            'lr': base_lr * ngpu,
            'labels': 'data-local/labels/cifar10/{}_balanced_labels/{:02d}.txt'.format(n_labels, data_seed),
    }
    # print(adapted_args)
    args = parse_dict_args(**adapted_args, **kwargs)
    return args

args = parse_parameters(**param)

dataset_config = datasets.__dict__['cifar10']()
num_classes = dataset_config.pop('num_classes')
train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
for i, ((inputs, ema_inputs), target) in enumerate(train_loader):
    print(inputs.shape)
    print(ema_inputs.shape)
    print(target)
    # break
    input()