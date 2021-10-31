import inspect

model_registry = {}


def registerModel(ctr, name=None):
    model_registry[ctr.__name__ if name is None else name] = ctr


class TmpCrt(object):
    def __init__(self, ctr, sub_args):
        self.ctr = ctr
        self.sub_args = sub_args

    def __call__(self, *args, **kwargs):
        # merge the two dictionaries, overriding with sub_args
        return self.ctr(*args, **{**kwargs, **self.sub_args})

# todo make this recursive


def buildModel(config=None, type=None, *pargs, **kwargs):
    if config is not None:
        return buildModel(**config)
    if 'args' in kwargs:
        kwargs.update(kwargs['args'])
        del kwargs['args']

    assert type in model_registry, "Unable to find {} in model registry {}".format(
        type, '\n'.join(model_registry.keys()))
    for k, v in kwargs.items():
        if isinstance(v, dict) and 'type' in v:
            tmp_type = v['type']
            assert tmp_type in model_registry, "Unable to find subtype {} in model registry {}".format(
                tmp_type, '\n'.join(model_registry.keys()))
            if 'args' in v:
                v = model_registry[tmp_type](**v['args'])
            else:
                v = model_registry[tmp_type]

            kwargs[k] = v
    return model_registry[type](**kwargs)


def completeConfig(config):
    typename = config['type']
    assert typename in model_registry, 'Unable to find {} in model registry {}'.format(
        typename, '\n'.join(model_registry.keys()))
    sig = inspect.signature(model_registry[typename].__init__)
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
