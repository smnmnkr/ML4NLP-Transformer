import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import build_vocab_from_iterator


class ImdbSentiment:

    def __init__(self):
        self.tokenizer = get_tokenizer('basic_english')

        self.train, self.test = IMDB(split=('train', 'test'))

        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.train), specials=['<unk>'])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def yield_tokens(self, data):
        for _, text in data:
            yield self.tokenizer(text)

    def text_pipeline(self, x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, y):
        return 1.0 if y == 'pos' else 0.0

    def collate_batch(self, batch):
        label_list, text_list = [], []

        for (_label, _text) in batch:

            label_list.append(self.label_pipeline(_label))
            text_list.append(torch.tensor(self.text_pipeline(_text)))

        return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)


if __name__ == "__main__":
    ydata = ImdbSentiment()

    print(ydata.train[0])

    train_dataloader = DataLoader(ydata.train, batch_size=1, collate_fn=ydata.collate_batch)

    for idx, batch in enumerate(train_dataloader):
        print(idx)
        print(batch)
