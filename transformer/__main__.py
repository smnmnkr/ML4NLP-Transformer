import argparse
import logging
import random
import time
from timeit import default_timer as timer

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformer.data.multi30k import Multi30KTranslation
from transformer.nn import Transformer
from transformer.tasks import train, evaluate, predict, save, load
from transformer.utils import EarlyStopping, get_device, load_json, save_json


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 description: str = "ML4NLP Seq2Seq Transformer",
                 data_cls=Multi30KTranslation):

        self.start_time: time = time.strftime("%Y%m%d-%H%M%S")
        self.description: str = description
        self.data_cls = data_cls

        self.config: dict = self.load_config()
        self.train_log: dict = {"epoch": [], "train_loss": [], "val_loss": []}

        # setup external libs
        self.setup_pytorch()
        self.setup_logging()

        # load data handler and model
        self.data = self.load_data()
        self.model = self.load_transformer()

        # load loss function, optimizer, scheduler, stopper
        self.loss_fn = CrossEntropyLoss(ignore_index=self.data.special_symbols['<pad>'])
        self.optimizer = Adam(self.model.parameters(), **self.config['training']['optimizer'])
        self.scheduler = ReduceLROnPlateau(self.optimizer, **self.config['training']['scheduler'])
        self.stopper = EarlyStopping(**self.config['training']['stopper'])

        # string format utils
        self.string_format: dict = {
            'epoch': lambda epoch: f"[{epoch}]",
            'loss': lambda name, val: f"loss({name}): {val:.3f}",
            'duration': lambda start, end: f"time: {(end - start):.3f}s"
        }

    #
    #
    #  -------- call -----------
    #
    def __call__(self):
        self.translate()
        self.train()
        self.translate()
        self.write_result()

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
    #  -------- setup_logging -----------
    #
    def setup_logging(self):
        filename: str = self.config['training']['log_path'] + self.start_time + ".log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] -- %(message)s",
            handlers=[
                logging.FileHandler(filename),
                logging.StreamHandler()
            ]
        )

    #
    #
    #  -------- load_data -----------
    #
    def load_data(self):
        logging.info("LOAD DATA")
        return self.data_cls()

    #
    #
    #  -------- load_transformer -----------
    #
    def load_transformer(self) -> Transformer:
        logging.info("LOAD TRANSFORMER")
        self.config['model']['src_vocab_size'] = len(self.data.vocab_transform['de'])
        self.config['model']['tgt_vocab_size'] = len(self.data.vocab_transform['en'])
        return Transformer(**self.config['model']).to(get_device())

    #
    #
    #  -------- translate -----------
    #
    def translate(self):
        logging.info("TRANSLATE")
        for idx, sent in enumerate(self.config['predict']):
            logging.info(f"[{idx}] {sent} -> {predict(self.model, self.data, sent)}")

    #
    #
    #  -------- train -----------
    #
    def train(self):
        logging.info("BEGIN TRAINING")
        try:
            for e in range(1, self.config['training']['epochs'] + 1):
                self.train_log["epoch"].append(e)

                # get data loaders (train, validation)
                train_loader = self.data.get_dataloader('train', batch_size=self.config['training']['batch_size'])
                val_loader = self.data.get_dataloader('valid', batch_size=self.config['training']['batch_size'])

                # train and measure time
                start_time = timer()
                self.train_log["train_loss"].append(train(self.model, self.optimizer, self.loss_fn, train_loader))
                end_time = timer()

                # evaluate and process scheduler and early stopping
                self.train_log["val_loss"].append(evaluate(self.model, self.loss_fn, val_loader))
                self.scheduler.step(self.train_log["val_loss"][-1])
                self.stopper.step(self.train_log["val_loss"][-1])

                if self.stopper.should_save:
                    self.save(e, self.train_log["train_loss"][-1], self.train_log["val_loss"][-1])

                if self.stopper.should_stop:
                    logging.info("Early stopping interrupted training")
                    break

                # append current data to log
                if e % self.config['training']['report_every'] == 0:
                    logging.info(self.string_format['epoch'](self.train_log["epoch"][-1]), '\t',
                                 self.string_format['loss']('train', self.train_log["train_loss"][-1]), '\t',
                                 self.string_format['loss']('val', self.train_log["val_loss"][-1]), '\t',
                                 self.string_format['duration'](start_time, end_time)
                                 )

        # handle user keyboard interruption
        except KeyboardInterrupt:
            logging.info("User interrupted training, trying to proceed and save model")

        # if exists load last saved model
        finally:
            try:
                info = self.load()
                logging.info("Loaded best model from checkpoint")
                logging.info(self.string_format['epoch'](info['epoch']), '\t',
                             self.string_format['loss']('train', info['train_loss']), '\t',
                             self.string_format['loss']('val', info['val_loss']),
                             )

            # use last internal model
            except FileNotFoundError:
                logging.info("No saved model found, last internal model")

    #
    #
    #  -------- save -----------
    #
    def save(self, epoch: int, train_loss: float, val_loss: float):
        logging.debug("Saving model to file")
        save(
            self.config['training']['log_path'] + self.start_time + '--model.pth',
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
    def load(self, path: str = None) -> dict:
        logging.debug("Loading model to file")

        if not path:
            path: str = self.config['training']['log_path'] + self.start_time + '--model.pth'

        self.model, info = load(path, Transformer)
        return info

    #
    #
    #  -------- write_result -----------
    #
    def write_result(self):
        path_log: str = self.config['training']['log_path'] + self.start_time + "--results.json"

        save_json(path_log, {
            "config": self.config,
            "train_log": self.train_log
        })


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()
