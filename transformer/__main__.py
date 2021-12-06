import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer.data.sentiment import SentimentCorpus
from transformer.nn import Transformer
from transformer.tasks import train, validate
from transformer.utils import get_device

if __name__ == "__main__":

    data = SentimentCorpus()
    model = Transformer(num_tokens=len(data.dictionary), num_outputs=1).to(get_device())

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    epochs: int = 10
    for epoch in range(epochs):
        train_loss = train(model, opt, loss_fn, DataLoader(data.train, batch_size=1))
        validation_loss = validate(model, loss_fn, DataLoader(data.valid, batch_size=1))

        print(f"[{epoch + 1:03d}] || loss(train): {train_loss:.4f} || loss(dev): {validation_loss:.4f}")
