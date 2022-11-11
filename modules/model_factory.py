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
    if inspect.isclass(name):
        return name
    if '.' in name:
        idx = name.rfind('.')
        module = name[0:idx]
        mod = importlib.import_module(module)
        name = name[idx+1:]
        # standard constructor
        ctr = eval(f'mod.{name}')
    else:
        ctr = eval(name)
    return ctr

object_dict = {}

def buildModel(config=None, type=None, ref=None, store: dict =None, path='./', *pargs, **kwargs):
    if store is not None:
        object_dict.update(store)
    if ref is not None:
        return object_dict[ref]
    if config is not None:
        if inspect.isclass(config):
            return buildModel(type=config, *pargs, **kwargs)
        else:
            return buildModel(**config, **kwargs)
    if 'args' in kwargs:
        kwargs.update(kwargs['args'])
        if 'pretrained' in kwargs['args']:
            pretrained = kwargs['args']['pretrained']
        del kwargs['args']

    save_as = None
    if 'as' in kwargs:
        save_as = kwargs['as']
        del kwargs['as']
    
    for k, v in kwargs.items():
        if isinstance(v, str) and '%' == v[0]:
            # perform value lookup
            v = v[1:]
            end = v.find('.')
            if end == -1:
                # no expression, we're accessing an object, not a sub object
                obj = object_dict[v]
                kwargs[k] = eval(f'obj')
            else:    
                obj_name = v[0:end]
                expression = v[v.find('.')+1:]
                obj = object_dict[obj_name]
                kwargs[k] = eval(f'obj.{expression}')

    for k, v in kwargs.items():
        if isinstance(v, dict) and 'type' in v:
            if 'args' in v:
                args = {}
                if v['args'] is not None:
                    args = v['args']
                new_v = buildModel(type=v['type'], **args)
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
    if inspect.isclass(ctr):
        return inspect.signature(ctr.__init__)
    else:
        return inspect.signature(ctr)
        

def recurseArguments(default_argument):
    if isinstance(default_argument, dict):
        output = {}
        for name, argument in default_argument.items():
            output[name] = recurseArguments(argument)
        return output
    if inspect.isclass(default_argument):
        class_name = repr(default_argument).split("'")[1]
        return {'type': class_name}
    else:
        return default_argument

def completeConfigForFunction(configuration_arguments, foo, allow_missing=False, typename = ''):
    sig = getSignature(foo)
    
    if configuration_arguments is None:
        configuration_arguments = {}
    for arg in sig.parameters.keys():
        if 'self' != arg and 'kwargs' != arg:
            if inspect._empty == sig.parameters[arg].default:
                if not allow_missing:
                    assert arg in configuration_arguments, "Required argument for {} '{}' not found in the provided configuration".format(typename,
                                                                                                                                      arg)
            else:
                # This is an optional argument, if it isn't in the config file, update it
                if not arg in configuration_arguments:
                    configuration_arguments[arg] = recurseArguments(sig.parameters[arg].default)
    return configuration_arguments

def completeConfig(config, allow_missing=False):
    typename = config['type']
    ctr = getConstructor(typename)
    configuration_arguments = config['args'] if 'args' in config else dict()
    config['args'] = completeConfigForFunction(configuration_arguments, ctr, typename=typename, allow_missing=allow_missing)
    return config
    

