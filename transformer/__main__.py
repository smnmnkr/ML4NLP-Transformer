from timeit import default_timer as timer

import torch

from transformer.data.multi30k import Multi30KTranslation
from transformer.nn import Transformer
from transformer.tasks import train, evaluate
from transformer.utils import get_device


if __name__ == "__main__":

    torch.manual_seed(0)

    print("[––– GENERATE VOCABS ---]")
    data_handler = Multi30KTranslation()

    print("[––– CREATING TRANSFORMER ---]")

    transformer_config: dict = {
        'num_encoder_layers': 2,
        'num_decoder_layers': 2,
        'emb_size': 128,
        'nhead': 4,
        'src_vocab_size': len(data_handler.vocab_transform['de']),
        'tgt_vocab_size': len(data_handler.vocab_transform['en']),
        'dim_feedforward': 128,
    }

    transformer = Transformer(**transformer_config)

    transformer = transformer.to(get_device())
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=data_handler.special_symbols['<pad>'])
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    print("[––– BEGIN TRAINING ---]")

    epochs: int = 10

    for e in range(1, epochs + 1):
        train_loader = data_handler.get_dataloader('train')
        val_loader = data_handler.get_dataloader('valid')

        start_time = timer()
        train_loss = train(transformer, optimizer, loss_fn, train_loader)
        end_time = timer()

        val_loss = evaluate(transformer, loss_fn, val_loader)

        print((
            f"[{e}], loss(train): {train_loss:.3f}, loss(val): {val_loss:.3f}, time: {(end_time - start_time):.3f}s"))
