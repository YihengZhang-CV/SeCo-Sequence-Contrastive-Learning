from model import resnet
import torch

model_dict = {
    'resnet18': resnet.resnet18,
    'resnet34': resnet.resnet34,
    'resnet50': resnet.resnet50,
    'resnet101': resnet.resnet101,
    'resnet103': resnet.resnet103,
    'resnet152': resnet.resnet152,
    'resnext50_32x4d': resnet.resnext50_32x4d,
    'resnext101_32x8d': resnet.resnext101_32x8d,
    'wide_resnet50_2': resnet.wide_resnet50_2,
    'wide_resnet101_2': resnet.wide_resnet101_2,
}


def get_model_by_name(net_name, **kwargs):
    return model_dict.get(net_name)(**kwargs)


def transfer_weights(net_name, state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        v = v.detach().numpy()
        new_state_dict[k] = torch.from_numpy(v)
    return new_state_dict


def remove_fc(net_name, state_dict):
    state_dict.pop('fc.weight', None)
    state_dict.pop('fc.bias', None)
    return state_dict
