from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator


class ImdbSentiment:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):

        # Define special symbols and indices
        self.special_symbols: dict = {
            '<unk>': 0,
            '<pad>': 1,
        }

        self.tokenizer = get_tokenizer('basic_english')

        self.tgt_vocab = build_vocab_from_iterator(iter([]), specials=['pos', 'neg'])
        self.src_vocab = build_vocab_from_iterator(
            self.yield_tokens(IMDB(split='train')),
            min_freq=1,
            specials=list(self.special_symbols.keys()),
            special_first=True
        )
        self.src_vocab.set_default_index(self.special_symbols['<unk>'])

    #
    #
    #  -------- yield_tokens -----------
    #
    def yield_tokens(self, data: Iterable):
        for _, text in data:
            yield self.tokenizer(text)

    #
    #
    #  -------- source_transform -----------
    #
    def source_transform(self, x: str) -> Tensor:
        return torch.tensor(self.src_vocab(self.tokenizer(x)))

    #
    #
    #  -------- target_transform -----------
    #
    def target_transform(self, y: str) -> Tensor:
        return torch.tensor(self.tgt_vocab([y]))

    #
    #
    #  -------- collate_fn -----------
    #
    def collate_fn(self, batch: list) -> Tuple[Tensor, Tensor]:
        """
        Function to collate data samples into batch tensors.

        :param batch:
        :return:
        """
        src_batch, tgt_batch = [], []

        for tgt_sample, src_sample in batch:
            src_batch.append(self.source_transform(src_sample))
            tgt_batch.append(self.target_transform(tgt_sample))

        src_batch = pad_sequence(src_batch, padding_value=self.special_symbols['<pad>'])

        return src_batch, tgt_batch

    #
    #
    #  -------- get_dataloader -----------
    #
    def get_dataloader(self, split: str = 'train', batch_size: int = 32) -> DataLoader:
        """
        Generate PyTorch DataLoader object

        :param split:
        :param batch_size:
        :return:
        """
        return DataLoader(IMDB(split=split), batch_size=batch_size, collate_fn=self.collate_fn)


if __name__ == "__main__":
    data = ImdbSentiment()
    train_dataloader = data.get_dataloader('train')

    for src, tgt in train_dataloader:
        print(src)
        exit()
