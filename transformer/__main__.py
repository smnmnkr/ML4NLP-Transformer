from typing import Any

import argparse
import random
import pprint
from timeit import default_timer as timer

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss

from transformer.data.multi30k import Multi30KTranslation
from transformer.nn import Transformer
from transformer.tasks import train, evaluate, predict, save, load
from transformer.utils import EarlyStopping, get_device, load_json


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 description: str = "ML4NLP Seq2Seq Transformer",
                 data_cls=Multi30KTranslation):

        self.description = description
        self.data_cls = data_cls
        self.config: dict = self.load_config()
        self.train_log: list = [['epoch', 'train_loss', 'val_loss']]

        self.setup_pytorch()
        self.data = self.load_data()
        self.model = self.load_transformer()

        # save config to log file
        self._write_log(self.config)

        # load loss function, optimizer, scheduler, stopper
        self.loss_fn = CrossEntropyLoss(ignore_index=self.data.special_symbols['<pad>'])
        self.optimizer = Adam(self.model.parameters(), **self.config['training']['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config['training']['scheduler'])
        self.stopper = EarlyStopping(**self.config['training']['stopper'])

        # string format utils
        self.string_format: dict = {
            'epoch': lambda epoch: f"[{epoch}]",
            'loss': lambda name, val: f"loss({name}): {val:.3f}",
            'duration': lambda start, end: f"\t time: {(end - start):.3f}s"
        }

    #
    #
    #  -------- call -----------
    #
    def __call__(self):
        self.translate()
        self.train()
        self.translate()

    #
    #
    #  -------- load_config -----------
    #
    def load_config(self) -> dict:
        # get console arguments, config file
        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument(
            "-C",
            dest="config",
            required=True,
            help="config.json file",
            metavar="FILE",
        )
        args = parser.parse_args()
        return load_json(args.config)

    #
    #
    #  -------- setup_pytorch -----------
    #
    def setup_pytorch(self):
        # make pytorch computations deterministic
        # src: https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(self.config['training']['seed'])
        torch.manual_seed(self.config['training']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self):
        print("[––– LOAD DATA ---]")
        return self.data_cls()

    #
    #
    #  -------- load_transformer -----------
    #
    def load_transformer(self) -> Transformer:
        print("[––– LOAD TRANSFORMER ---]")
        self.config['model']['src_vocab_size'] = len(self.data.vocab_transform['de'])
        self.config['model']['tgt_vocab_size'] = len(self.data.vocab_transform['en'])
        return Transformer(**self.config['model']).to(get_device())

    #
    #
    #  -------- translate -----------
    #
    def translate(self):
        print("[––– TRANSLATE ---]")
        for idx, sent in enumerate(self.config['predict']):
            print(f"[{idx}] {sent} -> {predict(self.model, self.data, sent)}")

    #
    #
    #  -------- train -----------
    #
    def train(self):
        print("[––– BEGIN TRAINING ---]")
        try:
            for e in range(1, self.config['training']['epochs'] + 1):

                # get data loaders (train, validation)
                train_loader = self.data.get_dataloader('train')
                val_loader = self.data.get_dataloader('valid')

                # train and measure time
                start_time = timer()
                train_loss = train(self.model, self.optimizer, self.loss_fn, train_loader)
                end_time = timer()

                # evaluate and process scheduler and early stopping
                val_loss = evaluate(self.model, self.loss_fn, val_loader)
                self.scheduler.step(val_loss)
                self.stopper.step(val_loss)

                if self.stopper.should_save:
                    self.save(e, train_loss, val_loss)

                if self.stopper.should_stop:
                    print("[––– Early stopping interrupted training ---]")
                    break

                # append current data to log, print according to report setting
                self.train_log.append([e, train_loss, val_loss])
                if e % self.config['training']['report_every'] == 0:
                    print(self.string_format['epoch'](e), '\t',
                          self.string_format['loss']('train', train_loss), '\t',
                          self.string_format['loss']('val', val_loss), '\t',
                          self.string_format['duration'](start_time, end_time)
                          )

        # handle user keyboard interruption
        except KeyboardInterrupt:
            print("[––– User interrupted training, trying to proceed and save model ---]")

        # write train log, and if exists load last saved model
        finally:
            self._write_log(self.train_log)

            try:
                info = self.load()
                print("[––– Loaded best model from checkpoint ---]")
                print(self.string_format['epoch'](info['epoch']), '\t',
                      self.string_format['loss']('train', info['train_loss']), '\t',
                      self.string_format['loss']('val', info['val_loss']), '\t'
                      )

            except FileNotFoundError:
                print("[––– No saved model found, last internal model ---]")

    #
    #
    #  -------- save -----------
    #
    def save(self, epoch: int, train_loss: float, val_loss: float):
        save(
            self.config['training']['log_path'] + 'model.pth',
            self.model,
            self.config['model'],
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss
        )

    #
    #
    #  -------- load -----------
    #
    def load(self) -> dict:
        path_model: str = self.config['training']['log_path'] + 'model.pth'

        self.model, info = load(path_model, Transformer)
        return info

    #
    #
    #  -------- _write_log -----------
    #
    def _write_log(self, content: Any):
        path_log: str = self.config['training']['log_path'] + "log.txt"

        with open(path_log, "a+") as log_file:
            pprint.pprint(content, log_file)


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
