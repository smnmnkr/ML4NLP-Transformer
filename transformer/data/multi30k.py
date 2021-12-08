from typing import Iterable, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import Vocab, build_vocab_from_iterator

from transformer.utils import sequential_transforms


class Multi30KTranslation:
    #
    #
    #  -------- __init__ -----------
    #
    def __init__(self):

        self.lang: dict = {
            'src': 'de',
            'tgt': 'en'
        }

        # Define special symbols and indices
        self.special_symbols: dict = {
            '<unk>': 0,
            '<pad>': 1,
            '<bos>': 2,
            '<eos>': 3
        }

        # Create source and target language tokenizer. Make sure to install the dependencies.
        # pip install -U spacy
        # python -m spacy download en_core_web_sm
        # python -m spacy download de_core_news_sm
        self.token_transform: dict = {
            self.lang['src']: get_tokenizer('spacy', language='de_core_news_sm'),
            self.lang['tgt']: get_tokenizer('spacy', language='en_core_web_sm'),
        }

        self.vocab_transform: dict = {
            ln: self.build_vocab(ln) for ln in list(self.lang.values())
        }

        # src and tgt language text transforms to convert raw strings into tensors indices
        self.text_transform: dict = {ln: sequential_transforms(
            self.token_transform[ln],  # Tokenization
            self.vocab_transform[ln],  # Numericalization
            self.tensor_transform,  # Add BOS/EOS and create tensor}
        ) for ln in list(self.lang.values())}

    #
    #
    #  -------- yield_tokens -----------
    #
    def yield_tokens(self, data: Iterable, language: str) -> List[str]:
        """
        Helper function to yield list of tokens

        :param data: Iterable dataset
        :param language: ('de'|'en')
        """
        lang_index: dict = {self.lang['src']: 0, self.lang['tgt']: 1}

        for sample in data:
            yield self.token_transform[language](sample[lang_index[language]])

    #
    #
    #  -------- build_vocab -----------
    #
    def build_vocab(self, language: str) -> Vocab:
        """
        Generate PyTorch Vocab object

        :rtype: Vocab
        """
        vocab = build_vocab_from_iterator(
            self.yield_tokens(Multi30k(split='train', language_pair=(self.lang['src'], self.lang['tgt'])), language),
            min_freq=1,
            specials=list(self.special_symbols.keys()),
            special_first=True
        )
        vocab.set_default_index(self.special_symbols['<unk>'])
        return vocab

    #
    #
    #  -------- tensor_transform -----------
    #
    def tensor_transform(self, token_ids: List[int]) -> Tensor:
        """
        Function to add BOS/EOS and create tensor for input sequence indices

        :param token_ids:
        :return: Tensor
        """
        return torch.cat((torch.tensor([self.special_symbols['<bos>']]),
                          torch.tensor(token_ids),
                          torch.tensor([self.special_symbols['<eos>']])))

    #
    #
    #  -------- tensor_transform -----------
    #
    def collate_fn(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Function to collate data samples into batch tensors.

        :param batch:
        :return:
        """
        src_batch, tgt_batch = [], []

        for src_sample, tgt_sample in batch:
            src_batch.append(self.text_transform[self.lang['src']](src_sample.rstrip("\n")))
            tgt_batch.append(self.text_transform[self.lang['tgt']](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=self.special_symbols['<pad>'])
        tgt_batch = pad_sequence(tgt_batch, padding_value=self.special_symbols['<pad>'])

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
        data = Multi30k(split=split, language_pair=(self.lang['src'], self.lang['tgt']))
        return DataLoader(data, batch_size=batch_size, collate_fn=self.collate_fn)
