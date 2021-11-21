from argparse import ArgumentParser
from modules.model_factory import buildModel, completeConfig
import pytorch_lightning as pl

import os
import yaml

class TrainingConfigurationManager(pl.Callback):
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--model', default='model.yaml')
        parser.add_argument('--hyp', default='hyp.yaml')
        parser.add_argument('--resume', default=None)
        args, unknown = parser.parse_known_args()
        self.model = self.loadDict(args.model)
        self.hyp = self.loadDict(args.hyp)
        if len(self.hyp) > 0:
            for k,v in self.hyp.items():
                parser.add_argument(f'--{k}', default=v)
            args = parser.parse_args()
        self.args = args

    def loadDict(self, path):
        return yaml.safe_load(open(path, 'rt'))

    def completeConfig(self):
        self.model = completeConfig(self.model)

    def saveConfig(self, path):
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, 'hyp.yaml'), 'wt') as f:
            yaml.dump(self.hyp, f)
        with open(os.path.join(path, 'model.yaml'), 'wt') as f:
            model = completeConfig(self.model)
            yaml.dump(model, f)

    def on_train_start(self, trainer, model):
        if hasattr(trainer.logger, 'log_dir'):
            self.saveConfig(trainer.logger.log_dir)