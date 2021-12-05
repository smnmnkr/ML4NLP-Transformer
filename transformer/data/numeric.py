import random

import numpy as np


#
#
#  -------- generate_random_numbers -----------
#
def generate_random_numbers(n: int = 9000) -> list:
    """
    Generates a list of random numeric samples

    :param n: amount
    :return: random numeric samples (list)
    """
    sos_token = np.array([2])
    eos_token = np.array([3])

    length: int = 8
    data: list = []

    # 1,1,1,1,1,1 -> 1,1,1,1,1
    for i in range(n // 3):
        x = np.concatenate((sos_token, np.ones(length), eos_token))
        y = np.concatenate((sos_token, np.ones(length), eos_token))
        data.append([x, y])

    # 0,0,0,0 -> 0,0,0,0
    for i in range(n // 3):
        x = np.concatenate((sos_token, np.zeros(length), eos_token))
        y = np.concatenate((sos_token, np.zeros(length), eos_token))
        data.append([x, y])

    # 1,0,1,0 -> 1,0,1,0,1
    for i in range(n // 3):
        x = np.zeros(length)
        start = random.randint(0, 1)

        x[start::2] = 1

        y = np.zeros(length)
        if x[-1] == 0:
            y[::2] = 1
        else:
            y[1::2] = 1

        x = np.concatenate((sos_token, x, eos_token))
        y = np.concatenate((sos_token, y, eos_token))

        data.append([x, y])

    np.random.shuffle(data)

    return data


#
#
#  -------- batchify_random_numbers -----------
#
def batchify_random_numbers(
        data,
        batch_size: int = 16,
        padding: bool = False,
        padding_token: int = -1,
) -> list:
    """
    Creates batches given the input data

    :param data: list of data
    :param batch_size: int
    :param padding: should pad data
    :param padding_token:
    :return: data batchloader
    """

    batches: list = []

    for idx in range(0, len(data), batch_size):

        # We make sure we dont get the last bit if its not batch_size size
        if idx + batch_size < len(data):

            # Here you would need to get the max length of the batch,
            # and normalize the length with the PAD token.
            if padding:
                max_batch_length = 0

                # Get longest sentence in batch
                for seq in data[idx: idx + batch_size]:
                    if len(seq) > max_batch_length:
                        max_batch_length = len(seq)

                # Append X padding tokens until it reaches the max length
                for seq_idx in range(batch_size):
                    remaining_length = max_batch_length - len(data[idx + seq_idx])
                    data[idx + seq_idx] += [padding_token] * remaining_length

            batches.append(np.array(data[idx: idx + batch_size]).astype(np.int64))

    return batches
