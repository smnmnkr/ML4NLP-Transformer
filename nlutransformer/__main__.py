import torch
from torch import nn

from nlutransformer.data.numeric import generate_random_numbers, batchify_random_numbers
from nlutransformer.nn import Transformer
from nlutransformer.tasks import train, validate
from nlutransformer.util import get_device


def fit(
        model: Transformer,
        opt,
        loss_fn,
        train_dataloader,
        val_dataloader,
        epochs
):
    for epoch in range(epochs):
        train_loss = train(model, opt, loss_fn, train_dataloader)
        validation_loss = validate(model, loss_fn, val_dataloader)

        print(f"[{epoch + 1:03d}] || loss(train): {train_loss:.4f} || loss(dev): {validation_loss:.4f}")


if __name__ == "__main__":
    train_data = generate_random_numbers(9000)
    val_data = generate_random_numbers(3000)

    train_dataloader = batchify_random_numbers(train_data)
    val_dataloader = batchify_random_numbers(val_data)

    model = Transformer(
        num_tokens=4,
        dim_model=8,
        num_heads=2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dropout=0.1
    ).to(get_device())

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    fit(model, opt, loss_fn, train_dataloader, val_dataloader, 10)
