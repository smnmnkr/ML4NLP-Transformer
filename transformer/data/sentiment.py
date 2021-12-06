from datasets import load_dataset, Dataset

from transformer.utils import Dictionary


class SentimentCorpus:
    def __init__(self):
        self.dictionary = Dictionary()

        self.train = self.get_data('train')
        self.valid = self.get_data('validation')
        self.test = self.get_data('test')

    def get_data(self, split: str = 'train') -> Dataset:
        dataset = load_dataset('sst', 'default', split=split)

        dataset = dataset.map(
            lambda entry: {'token_ids': [self.dictionary.add_word(token) for token in entry['tokens'].split("|")]})

        dataset.set_format(type='torch', columns=['label', 'token_ids'])

        return dataset


if __name__ == "__main__":
    testCorpus = SentimentCorpus()
    print(len(testCorpus.dictionary))
