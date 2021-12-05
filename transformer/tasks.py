import torch

from transformer.util import get_device, get_tgt_mask


#
#
#  -------- train -----------
#
def train(model, opt, loss_fn, dataloader) -> float:
    """
    Trains a model using opt and loss_fn on given dataloader

    :param model: transformer model
    :param opt: pyTorch optimizer
    :param loss_fn: pyTorch loss function
    :param dataloader: batched data loader
    :return: loss value (float)
    """
    model.train()
    total_loss: float = 0

    for batch in dataloader:
        loss, _ = _step(model, loss_fn, batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)


#
#
#  -------- validate -----------
#
def validate(model, loss_fn, dataloader) -> float:
    """
    Validates a model using loss_fn on given dataloader

    :param model: transformer model
    :param loss_fn: pyTorch loss function
    :param dataloader: batched data loader
    :return: loss value (float)
    """
    model.eval()
    total_loss: float = 0

    with torch.no_grad():
        for batch in dataloader:
            loss, _ = _step(model, loss_fn, batch)
            total_loss += loss.detach().item()

    return total_loss / len(dataloader)


#
#
#  -------- _step -----------
#
def _step(model, loss_fn, batch):
    """
    Internal data processing and computing function used in train, validate.

    :param model: transformer model
    :param loss_fn: pyTorch loss function
    :param batch: data batch
    :return: loss tensor, models prediction
    """
    x, y = batch[:, 0], batch[:, 1]
    x, y = torch.tensor(x).to(get_device()), torch.tensor(y).to(get_device())

    # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
    y_input = y[:, :-1]
    y_expected = y[:, 1:]

    # Get mask to mask out the next words
    sequence_length = y_input.size(1)
    tgt_mask = get_tgt_mask(sequence_length).to(get_device())

    # Standard training except we pass in y_input and tgt_mask
    pred = model(x, y_input, tgt_mask)

    # Permute pred to have batch size first again
    pred = pred.permute(1, 2, 0)
    loss = loss_fn(pred, y_expected)

    return loss, pred
