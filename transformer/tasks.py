from typing import Tuple, Dict, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from transformer.utils import get_device, create_mask, generate_square_subsequent_mask


#
#
#  -------- train -----------
#
def train(
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        loss_fn,
        train_dataloader: DataLoader) -> float:
    model.train()
    losses: float = 0

    for src, tgt in train_dataloader:
        tgt, logits = _step(src, tgt, model)

        optim.zero_grad()

        tgt_out: Tensor = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optim.step()
        losses += loss.item()

    return losses / len(train_dataloader)


#
#
#  -------- evaluate -----------
#
def evaluate(model: torch.nn.Module, loss_fn, val_dataloader: DataLoader) -> float:
    model.eval()
    losses: float = 0

    for src, tgt in val_dataloader:
        tgt, logits = _step(src, tgt, model)

        tgt_out: Tensor = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


#
#
#  -------- predict -----------
#
def predict(model: torch.nn.Module, data_handler, src_sentence: str) -> str:
    model.eval()

    src: Tensor = data_handler.text_transform[data_handler.lang['src']](src_sentence).view(-1, 1)

    num_tokens: int = src.shape[0]
    src_mask: Tensor = (torch.zeros(num_tokens, num_tokens))

    tgt_tokens = _greedy_decode(model, src, src_mask, max_len=num_tokens + 5,
                                start_symbol=data_handler.special_symbols['<bos>'],
                                end_symbol=data_handler.special_symbols['<eos>']).flatten()

    return " ".join(
        data_handler.vocab_transform[data_handler.lang['tgt']].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(
        "<bos>", "").replace("<eos>", "")


#
#
#  -------- save -----------
#
def save(path: str,
         model: torch.nn.Module,
         model_config: dict,
         epoch: int = 0,
         train_loss: float = 0,
         val_loss: float = 0) -> None:
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss
    }, path)


#
#
#  -------- load -----------
#
def load(path: str, model_cls) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    checkpoint: dict = torch.load(path)
    model: torch.nn.Module = model_cls(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])

    return (model, {
        'epoch': checkpoint['epoch'],
        'train_loss': checkpoint['train_loss'],
        'val_loss': checkpoint['val_loss'],
    })


#
#
#  -------- _step -----------
#
def _step(src: Tensor, tgt: Tensor, model: torch.nn.Module, pad_idx: int = 1):
    src = src.to(get_device())
    tgt = tgt.to(get_device())

    tgt_input = tgt[:-1, :]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, pad_idx)

    logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

    return tgt, logits


#
#
#  -------- _greedy_decode -----------
#
def _greedy_decode(
        model: torch.nn.Module,
        src: Tensor,
        src_mask: Tensor,
        max_len: int,
        start_symbol: str,
        end_symbol: str
) -> Tensor:
    src = src.to(get_device())
    src_mask = src_mask.to(get_device())

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(get_device())

    for i in range(max_len - 1):
        memory = memory.to(get_device())
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))).to(get_device())

        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)

        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == end_symbol:
            break

    return ys
