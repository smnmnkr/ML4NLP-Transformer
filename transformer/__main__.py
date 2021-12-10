import argparse
from timeit import default_timer as timer
import pprint
from typing import Any

import torch
import random

from transformer.data.multi30k import Multi30KTranslation
from transformer.nn import Transformer
from transformer.tasks import train, evaluate, predict, save
from transformer.utils import get_device, load_json


class Main:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self,
                 description: str = "ML4NLP Seq2Seq Transformer",
                 data_obj=Multi30KTranslation):

        self.description = description
        self.data_obj = data_obj
        self.config: dict = self.load_config()

        self.setup_pytorch()
        self.data = self.load_data()
        self.model = self.load_transformer()

        # save config to log file
        self._write_log(self.config)

        # load loss function, optimizer, scheduler
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.data.special_symbols['<pad>'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.config['training']['optimizer'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    **self.config['training']['scheduler'])

    #
    #
    #  -------- call -----------
    #
    def __call__(self):
        self.translate()
        self.train()
        self.translate()
        self.save()

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
        return self.data_obj()

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

        #  -------- format_epoch -----------
        #
        def format_epoch():
            return ((
                f"[{e}]"
                f"\t loss(train): {train_loss:.3f}"
                f"\t loss(val): {val_loss:.3f}"
                f"\t time: {duration:.3f}s"
            ))

        try:
            for e in range(1, self.config['training']['epochs'] + 1):
                train_loader = self.data.get_dataloader('train')
                val_loader = self.data.get_dataloader('valid')

                start_time = timer()
                train_loss = train(self.model, self.optimizer, self.loss_fn, train_loader)
                end_time = timer()
                duration: float = end_time - start_time

                val_loss = evaluate(self.model, self.loss_fn, val_loader)
                self.scheduler.step(val_loss)

                self._write_log(format_epoch())
                if e % self.config['training']['report_every'] == 0:
                    print(format_epoch())

        # handle user keyboard interruption
        except KeyboardInterrupt:
            print("[––– User interrupted training, trying to proceed and save model ---]")

    #
    #
    #  -------- save -----------
    #
    def save(self):
        print("[––– SAVING MODEL ---]")
        save(self.config['training']['log_path'] + 'model.pth', self.model, self.config['model'])

    #
    #
    #  -------- _write_log -----------
    #
    def _write_log(self, content: Any):
        with open(self.config['training']['log_path'] + "train.log", "w") as log_file:
            pprint.pprint(content, log_file)


#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":
    Main()()

