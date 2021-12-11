# Transformer Example for ML4NLP - Winter 21/22

## Usage
```bash
# download the repository to your local machine
git clone https://github.com/smnmnkr/ML4NLP-Transformer.git

# move into repository
cd ML4NLP-Transformer

# install requirements
make install

# or install an conda environment
conda env create -f venv.yml

# run default configuration
# if you use conda activate the environment first:
# conda activate Seq2SeqTransformer
make run
```

## Example Log (based on config.json)
```terminal
[––– LOAD DATA ---]
[––– LOAD TRANSFORMER ---]
[––– TRANSLATE ---]
[0] ein mann steht auf einem baugerüst . ->  centipede centipede centipede centipede centipede centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[1] männer spielen auf trommeln . ->  centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[2] ein kind läuft auf einem weg . ->  centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[––– BEGIN TRAINING ---]
[5]      loss(train): 3.180      loss(val): 3.103        time: 43.225s
[10]     loss(train): 2.392      loss(val): 2.407        time: 43.085s
[15]     loss(train): 1.928      loss(val): 2.135        time: 45.843s
[20]     loss(train): 1.647      loss(val): 2.018        time: 43.312s
[25]     loss(train): 1.454      loss(val): 1.976        time: 42.599s
[30]     loss(train): 1.305      loss(val): 1.963        time: 41.974s
[35]     loss(train): 1.183      loss(val): 1.993        time: 45.915s
[40]     loss(train): 1.066      loss(val): 1.968        time: 47.431s
[45]     loss(train): 0.981      loss(val): 1.985        time: 45.653s
[50]     loss(train): 0.897      loss(val): 2.002        time: 42.314s
[––– Early stopping interrupted training ---]
[––– Loaded best model from checkpoint ---]
[27]     loss(train): 1.393      loss(val): 1.958
[––– TRANSLATE ---]
[0] ein mann steht auf einem baugerüst . ->  A classroom of people standing up on a tiled platform . 
[1] männer spielen auf trommeln . ->  LeBron students are playing the drums . 
[2] ein kind läuft auf einem weg . ->  A kid is running away from a cliff .   
```

## Example Config (reduced network size)
```json
{
  "model": {
    "num_encoder_layers": 2,
    "num_decoder_layers": 2,
    "emb_size": 128,
    "nhead": 4,
    "dim_feedforward": 128,
    "dropout": 0.3
  },
  "training": {
    "seed": 42,
    "epochs": 40,
    "report_every": 5,
    "log_path": "./results/XX/",
    "optimizer": {
      "lr": 1e-4,
      "betas": [
        0.9,
        0.98
      ],
      "eps": 1e-9
    },
    "scheduler": {
      "factor": 0.8,
      "patience": 10
    },
    "stopper": {
      "delta": 0.1,
      "patience": 20
    }
  },
  "predict": [
    "ein mann steht auf einem baugerüst .",
    "männer spielen auf trommeln .",
    "ein kind läuft auf einem weg ."
  ]
}
```

## Sources

* PyTorch Transformer, "NLP From Scratch": <https://pytorch.org/tutorials/beginner/translation_transformer.html>
* A detailed guide to PyTorch’s nn.Transformer() module: <https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1>