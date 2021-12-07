# Transformer Example for ML4NLP - Winter 21/22

## Usage
```bash
# download the repository to your local machine
git clone https://github.com/smnmnkr/ML4NLP-Transformer.git

# move into repository
cd ML4NLP-Transformer

# install requirements
make install

# run default configuration
make run
```

## Example Log
```terminal
[––– GENERATE VOCABS ---]
[––– CREATING TRANSFORMER ---]
[––– BEGIN TRAINING ---]
[2] 	 loss(train): 4.122 	 loss(val): 3.770 	 time: 260.780s
[4] 	 loss(train): 3.383 	 loss(val): 3.170 	 time: 462.554s
[6] 	 loss(train): 2.952 	 loss(val): 2.803 	 time: 452.415s
[8] 	 loss(train): 2.647 	 loss(val): 2.558 	 time: 300.493s
[10] 	 loss(train): 2.411 	 loss(val): 2.392 	 time: 265.002s
[12] 	 loss(train): 2.221 	 loss(val): 2.275 	 time: 268.233s
[14] 	 loss(train): 2.062 	 loss(val): 2.187 	 time: 265.326s
[16] 	 loss(train): 1.929 	 loss(val): 2.122 	 time: 267.503s
[18] 	 loss(train): 1.815 	 loss(val): 2.077 	 time: 267.203s
[20] 	 loss(train): 1.718 	 loss(val): 2.042 	 time: 272.205s
[22] 	 loss(train): 1.633 	 loss(val): 2.019 	 time: 266.967s
[24] 	 loss(train): 1.557 	 loss(val): 2.001 	 time: 266.322s
[26] 	 loss(train): 1.490 	 loss(val): 1.995 	 time: 267.105s
[28] 	 loss(train): 1.429 	 loss(val): 1.983 	 time: 271.925s
[30] 	 loss(train): 1.374 	 loss(val): 1.981 	 time: 261.512s
[32] 	 loss(train): 1.324 	 loss(val): 1.976 	 time: 273.604s
[34] 	 loss(train): 1.279 	 loss(val): 1.967 	 time: 268.140s
[36] 	 loss(train): 1.238 	 loss(val): 1.983 	 time: 245.439s
[38] 	 loss(train): 1.199 	 loss(val): 1.974 	 time: 277.088s
[40] 	 loss(train): 1.163 	 loss(val): 1.981 	 time: 280.415s
```

## Example Config
```json
{
  "model": {
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "emb_size": 128,
    "nhead": 4,
    "dim_feedforward": 128
  },
  "optim": {
    "lr": 1e-4,
    "betas": [0.9, 0.98],
    "eps": 1e-9
  },
  "predict": [
    "ein mann steht auf einem baugerüst .",
    "männer spielen auf trommeln .",
    "ein kind läuft auf einem weg ."
  ],
  "epochs": 40
}
```

## Sources

* PyTorch Transformer, "NLP From Scratch": https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html>
* A detailed guide to PyTorch’s nn.Transformer() module: <https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1>