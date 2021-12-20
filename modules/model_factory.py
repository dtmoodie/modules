import inspect
import importlib
import json
import os

import torch
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
try:
    from torch.nn.modules.channelshuffle import *
except:
    pass

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
        if 'timm' in module:
            def timmFactoryHelper(pretrained=False, checkpoint_path=None, scriptable=False, exportable=False, no_jit=False, drop_rate=0.0, **kwargs):
                return mod.create_model(name, 
                                        pretrained=pretrained, 
                                        checkpoint_path=checkpoint_path, 
                                        scriptable=scriptable, 
                                        exportable=exportable, 
                                        no_jit=no_jit, 
                                        drop_rate=drop_rate, 
                                        **kwargs)
            # forward stuff to timm factory helper
            ctr = timmFactoryHelper
        else:
            # standard constructor
            ctr = eval(f'mod.{name}')
    else:
        ctr = eval(name)
    return ctr

object_dict = {}

def buildModel(config=None, type=None, ref=None, path='./', pretrained=False, *pargs, **kwargs):
    if ref is not None:
        return object_dict[ref]
    if config is not None:
        return buildModel(**config)
    if 'args' in kwargs:
        kwargs.update(kwargs['args'])
        if 'pretrained' in kwargs['args']:
            pretrained = kwargs['args']['pretrained']
        del kwargs['args']

    if pretrained:
        # TODO testing since we do some manipulation above
        config = {'type': type, **kwargs}
        model_hash = hash(json.dumps(config, sort_keys=True))
        path_ = os.path.join(path, str(model_hash) + '.pt')
        if os.path.exists(path_):
            return torch.load(path_)

    save_as = None
    if 'as' in kwargs:
        save_as = kwargs['as']
        del kwargs['as']
    
    for k, v in kwargs.items():
        if isinstance(v, str) and '%' == v[0]:
            # perform value lookup
            v = v[1:]
            obj_name = v[0:v.find('.')]
            expression = v[v.find('.')+1:]
            obj = object_dict[obj_name]
            kwargs[k] = eval(f'obj.{expression}')

    for k, v in kwargs.items():
        if isinstance(v, dict) and 'type' in v:
            if 'args' in v:
                new_v = buildModel(type=v['type'], **v['args'])
                if 'as' in v:
                    object_dict[v['as']] = new_v

            else:
                ctr = getConstructor(v['type'])
                if len(v) > 1:
                    tmp_args = dict(v)
                    del tmp_args['type']
                    class Helper:
                        def __init__(self, ctr):
                            self.ctr = ctr
                        def __call__(self, *pargs, **kwargs):
                            return self.ctr(*pargs, **kwargs)
                    ctr = Helper(ctr)
                
                new_v = ctr

            kwargs[k] = new_v
    ctr = getConstructor(name=type)
    obj = ctr(**kwargs)
    if save_as is not None:
        object_dict[save_as] = obj
    return obj


def cacheModel(config, model, path='./'):
    model_hash = hash(json.dumps(config, sort_keys=True))
    torch.save(model, os.path.join(path, str(model_hash) + '.pt'))

def getSignature(ctr):
    if callable(ctr):
        return inspect.signature(ctr)
    else:
        return inspect.signature(ctr.__init__)

def completeConfig(config):
    typename = config['type']
    ctr = getConstructor(typename)
    sig = getSignature(ctr)
    configuration_arguments = config['args'] if 'args' in config else dict()
    if configuration_arguments is None:
        configuration_arguments = {}
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

