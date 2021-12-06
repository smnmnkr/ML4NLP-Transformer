class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def get_idx(self, word: str) -> int:
        return self.word2idx.get(word, default=-1)

    def __len__(self):
        return len(self.idx2word)
