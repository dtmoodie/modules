from argparse import ArgumentParser
from modules.model_factory import buildModel, completeConfig, completeConfigForFunction
import pytorch_lightning as pl
import torch

import os
import yaml

class GenericDataModule(pl.LightningDataModule):
    def __init__(self, train, val=None, test=None, dataloader_config={}):
        self.train = train
        self.val = val
        if val is None:
            self.val = train
        self.test = test
        if test is None:
            self.test = train
        self.dataloader_config = dataloader_config

    def createLoader(self, ds):
        return torch.utils.data.DataLoader(dataset=ds, **self.dataloader_config)

    def train_dataloader(self):
        return self.createLoader(self.train)

    def val_dataloader(self):
        return self.createLoader(self.val)

    def test_dataloader(self):
        return self.createLoader(self.train)    


class TrainingConfigurationManager(pl.Callback):
    def __init__(self, **kwargs):
        parser = ArgumentParser()
        parser.add_argument('--model', default='model.yaml')
        parser.add_argument('--hyp', default='hyp.yaml')
        parser.add_argument('--resume', default=None)
        for k,v in kwargs.items():
            if isinstance(v, bool):
                parser.add_argument('--' + k, action='store_true')
            else:
                parser.add_argument('--' + k, default=v)
        args, unknown = parser.parse_known_args()
        self.model = self.loadDict(args.model)
        self.hyp = self.loadDict(args.hyp)
        if len(self.hyp) > 0:
            for k,v in self.hyp.items():
                parser.add_argument(f'--{k}', default=v)
            args = parser.parse_args()
        self.args = args
        self.hyper_parameter_handlers = {}
        self.setHyperParameterHandler('trainer', pl.Trainer)

    def loadDict(self, path):
        return yaml.safe_load(open(path, 'rt'))

    def setHyperParameterHandler(self, section_name, handler):
        self.hyper_parameter_handlers[section_name] = handler

    def completeConfig(self, allow_missing=True):
        self.model = completeConfig(self.model, allow_missing=allow_missing)
        for k,v in self.hyper_parameter_handlers.items():
            self.hyp[k] = completeConfigForFunction(self.hyp[k], v, allow_missing=allow_missing)
        
    def saveConfig(self, path, allow_missing=True):
        self.completeConfig(allow_missing=True)
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'hyp.yaml'), 'wt') as f:
            yaml.dump(self.hyp, f)

        with open(os.path.join(path, 'model.yaml'), 'wt') as f:
            model = completeConfig(self.model, allow_missing=allow_missing)
            yaml.dump(model, f)

    def on_train_start(self, trainer, model):
        if trainer.global_rank == 0:
            if hasattr(trainer.logger, 'log_dir'):
                self.saveConfig(trainer.logger.log_dir)

    def createDataset(self, data_factory, dataset_args = {}, dataloader_args = {}):
        args = self.hyp['dataset']
        if 'dataloader' in self.hyp:
            dataloader_args.update(self.hyp['dataloader'])
        if 'subsets' in args:
            subsets = args['subsets']
            del args['subsets']
            subs = {}
            for name, subset_args in subsets.items():
                subs[name] = data_factory(**subset_args, **args, **dataset_args)
            return GenericDataModule(**subs, dataloader_config=dataloader_args)
        return GenericDataModule(train=data_factory(**args, **dataset_args), dataloader_config=dataloader_args)

    def createTrainer(self, **kwargs):
        args = dict(**self.hyp['trainer'])
        args.update(kwargs)
        return pl.Trainer(**args)

    
