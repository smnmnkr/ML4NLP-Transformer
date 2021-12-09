import argparse
from timeit import default_timer as timer

import torch
import random

from transformer.data.multi30k import Multi30KTranslation
from transformer.nn import Transformer
from transformer.tasks import train, evaluate, predict
from transformer.utils import get_device, load_json


# make pytorch computations deterministic
# src: https://pytorch.org/docs/stable/notes/randomness.html
random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#
#
#  -------- argparse -----------
#
parser = argparse.ArgumentParser(description="ML4NLP Seq2Seq Transformer")
parser.add_argument(
    "-C",
    dest="config",
    required=True,
    help="config.json file",
    metavar="FILE",
)

#
#
#  -------- __main__ -----------
#
if __name__ == "__main__":

    # get console arguments, config file
    args = parser.parse_args()
    config: dict = load_json(args.config)

    print("[––– GENERATE VOCABS ---]")
    data_handler = Multi30KTranslation()

    print("[––– CREATING TRANSFORMER ---]")
    transformer = Transformer(
        src_vocab_size=len(data_handler.vocab_transform['de']),
        tgt_vocab_size=len(data_handler.vocab_transform['en']),
        **config['model']
    ).to(get_device())

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_handler.special_symbols['<pad>'])
    optimizer = torch.optim.Adam(transformer.parameters(), **config['optim'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    print("[––– TRANSLATE (before train) ---]")
    for idx, sent in enumerate(config['predict']):
        print(f"[{idx}] {sent} -> {predict(transformer, data_handler, sent)}")

    print("[––– BEGIN TRAINING ---]")
    for e in range(1, config['epochs'] + 1):
        train_loader = data_handler.get_dataloader('train')
        val_loader = data_handler.get_dataloader('valid')

        start_time = timer()
        train_loss = train(transformer, optimizer, loss_fn, train_loader)
        end_time = timer()

        val_loss = evaluate(transformer, loss_fn, val_loader)
        scheduler.step(val_loss)

        if config['epochs'] + 1 % config['report_every'] == 0:
            print((
                f"[{e}] \t loss(train): {train_loss:.3f} \t loss(val): {val_loss:.3f} \t time: {(end_time - start_time):.3f}s"))

    print("[––– TRANSLATE (after train) ---]")
    for idx, sent in enumerate(config['predict']):
        print(f"[{idx}] {sent} -> {predict(transformer, data_handler, sent)}")