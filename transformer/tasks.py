import torch
from torch import Tensor

from transformer.utils import get_device, create_mask, generate_square_subsequent_mask


#
#
#  -------- train -----------
#
def train(model: torch.nn.Module, optim, loss_fn, train_dataloader):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        tgt, logits = _step(src, tgt, model)

        optim.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optim.step()
        losses += loss.item()

    return losses / len(train_dataloader)


#
#
#  -------- evaluate -----------
#
def evaluate(model: torch.nn.Module, loss_fn, val_dataloader):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        tgt, logits = _step(src, tgt, model)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


#
#
#  -------- predict -----------
#
def predict(model: torch.nn.Module, data_handler, src_sentence: str):
    model.eval()

    src: Tensor = data_handler.text_transform[data_handler.lang['src']](src_sentence).view(-1, 1)

    num_tokens = src.shape[0]
    src_mask: Tensor = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)

    tgt_tokens = _greedy_decode(model, src, src_mask, max_len=num_tokens + 5,
                                start_symbol=data_handler.special_symbols['<bos>'],
                                end_symbol=data_handler.special_symbols['<eos>']).flatten()

    return " ".join(
        data_handler.vocab_transform[data_handler.lang['tgt']].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace(
        "<bos>", "").replace("<eos>", "")


#
#
#  -------- _step -----------
#
def _step(src, tgt, model: torch.nn.Module, pad_idx: int = 1):
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
):
    src = src.to(get_device())
    src_mask = src_mask.to(get_device())

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(get_device())

    for i in range(max_len - 1):
        memory = memory.to(get_device())
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(get_device())

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
