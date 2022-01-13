from collections import namedtuple
from models.modeling import BasicBlock, Bottleneck, get_vgg_layers
import ml_collections

"""VGG"""
def vgg11_config():
    vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return get_vgg_layers(vgg11_config, batch_norm = True)

def vgg13_config():
    vgg13_config =  [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return get_vgg_layers(vgg13_config, batch_norm = True)

def vgg16_config():
    vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return get_vgg_layers(vgg16_config, batch_norm = True)

def vgg19_config():
    vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    return get_vgg_layers(vgg19_config, batch_norm = True)


"""ResNet"""
ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
def resnet18_config():
    return ResNetConfig(block = BasicBlock,
                        n_blocks = [2, 2, 2, 2],
                        channels = [64, 128, 256, 512])

def resnet34_config():
    return ResNetConfig(block = BasicBlock,
                        n_blocks = [3, 4, 6, 3],
                        channels = [64, 128, 256, 512])

def resnet50_config():
    return ResNetConfig(block = Bottleneck,
                        n_blocks = [3, 4, 6, 3],
                        channels = [64, 128, 256, 512])

def resnet101_config():
    return ResNetConfig(block = Bottleneck,
                        n_blocks = [3, 4, 23, 3],
                        channels = [64, 128, 256, 512])

def resnet151_config():
    return ResNetConfig(block = Bottleneck,
                        n_blocks = [3, 8, 36, 3],
                        channels = [64, 128, 256, 512])

"""TransFG"""
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.split = 'non-overlap'
    config.slide_step = 12
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config



CONFIGS = {
    'vgg11': vgg11_config(),
    'vgg13': vgg13_config(),
    'vgg16': vgg16_config(),
    'vgg19': vgg19_config(),
    'resnet18': resnet18_config(),
    'resnet34': resnet34_config(),
    'resnet50': resnet50_config(),
    'resnet101': resnet101_config(),
    'resnet151': resnet151_config(),
    'googlenet': None,
    'inceptionv3': None, 
    'transfg': get_b16_config()
}




