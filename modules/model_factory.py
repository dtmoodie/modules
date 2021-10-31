import inspect
import importlib

from torch.nn.modules.activation import *
from torch.nn.modules.linear import *
from torch.nn.modules.conv import *
from torch.nn.modules.container import *
from torch.nn.modules.batchnorm import *
from torch.nn.modules.pooling import *
from torch.nn.modules.instancenorm import *
from torch.nn.modules.normalization import *
from torch.nn.modules.dropout import *
from torch.nn.modules.upsampling import *
from torch.nn.modules.distance import *
from torch.nn.modules.adaptive import *
from torch.nn.modules.transformer import *
from torch.nn.modules.flatten import *
from torch.nn.modules.channelshuffle import *

class TmpCrt(object):
    def __init__(self, ctr, sub_args):
        self.ctr = ctr
        self.sub_args = sub_args

    def __call__(self, *args, **kwargs):
        # merge the two dictionaries, overriding with sub_args
        return self.ctr(*args, **{**kwargs, **self.sub_args})

def getConstructor(name):
    if '.' in name:
        idx = name.rfind('.')
        module = name[0:idx]
        mod = importlib.import_module(module)
        name = name[idx+1:]
        ctr = eval(f'mod.{name}')
    else:
        ctr = eval(name)
    return ctr

# TODO make this recursive
def buildModel(config=None, type=None, *pargs, **kwargs):
    if config is not None:
        return buildModel(**config)
    if 'args' in kwargs:
        kwargs.update(kwargs['args'])
        del kwargs['args']

    for k, v in kwargs.items():
        if isinstance(v, dict) and 'type' in v:
            if 'args' in v:
                v = buildModel(**v['args'])
            else:
                ctr = getConstructor(v['type'])
                v = ctr

            kwargs[k] = v
    ctr = getConstructor(name=type)
    return ctr(**kwargs)


def completeConfig(config):
    typename = config['type']
    ctr = getConstructor(typename)
    sig = inspect.signature(ctr.__init__)
    configuration_arguments = config['args'] if 'args' in config else dict()
    for arg in sig.parameters.keys():
        if 'self' != arg and 'kwargs' != arg:
            if inspect._empty == sig.parameters[arg].default:
                assert arg in configuration_arguments, "Required argument for {} '{}' not found in the provided configuration".format(typename,
                                                                                                                                      arg)
            else:
                # This is an optional argument, if it isn't in the config file, update it
                if not arg in configuration_arguments:
                    configuration_arguments[arg] = sig.parameters[arg].default
    config['args'] = configuration_arguments
    return config

