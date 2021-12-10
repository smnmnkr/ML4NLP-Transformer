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
[––– GENERATE VOCABS ---]
[––– CREATING TRANSFORMER ---]
[––– TRANSLATE (before train) ---]
[––– GENERATE VOCABS ---]
[––– CREATING TRANSFORMER ---]
[––– TRANSLATE (before train) ---]
[0] ein mann steht auf einem baugerüst . ->  centipede centipede centipede centipede centipede centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[1] männer spielen auf trommeln . ->  centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[2] ein kind läuft auf einem weg . ->  centipede sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk sidwalk
[––– BEGIN TRAINING ---]
[1]      loss(train): 4.868      loss(val): 4.090        time: 42.933s
[2]      loss(train): 3.909      loss(val): 3.692        time: 42.885s
[3]      loss(train): 3.582      loss(val): 3.429        time: 42.861s
[4]      loss(train): 3.357      loss(val): 3.264        time: 42.822s
[5]      loss(train): 3.180      loss(val): 3.103        time: 42.755s
[6]      loss(train): 3.006      loss(val): 2.906        time: 42.731s
[7]      loss(train): 2.824      loss(val): 2.760        time: 42.692s
[8]      loss(train): 2.679      loss(val): 2.636        time: 43.013s
[9]      loss(train): 2.526      loss(val): 2.532        time: 42.750s
[10]     loss(train): 2.392      loss(val): 2.407        time: 42.856s
[11]     loss(train): 2.274      loss(val): 2.323        time: 42.908s
[12]     loss(train): 2.171      loss(val): 2.270        time: 42.788s
[13]     loss(train): 2.079      loss(val): 2.205        time: 42.857s
[14]     loss(train): 1.997      loss(val): 2.172        time: 42.744s
[15]     loss(train): 1.928      loss(val): 2.135        time: 42.865s
[16]     loss(train): 1.860      loss(val): 2.098        time: 42.719s
[17]     loss(train): 1.802      loss(val): 2.075        time: 42.710s
[18]     loss(train): 1.745      loss(val): 2.045        time: 42.770s
[19]     loss(train): 1.696      loss(val): 2.034        time: 42.695s
[20]     loss(train): 1.647      loss(val): 2.018        time: 42.666s
[21]     loss(train): 1.605      loss(val): 2.018        time: 42.730s
[22]     loss(train): 1.563      loss(val): 2.011        time: 42.631s
[23]     loss(train): 1.525      loss(val): 1.997        time: 42.645s
[24]     loss(train): 1.489      loss(val): 1.997        time: 42.657s
[25]     loss(train): 1.454      loss(val): 1.976        time: 42.280s
[26]     loss(train): 1.422      loss(val): 1.985        time: 42.539s
[27]     loss(train): 1.393      loss(val): 1.958        time: 42.637s
[28]     loss(train): 1.360      loss(val): 1.962        time: 42.426s
[29]     loss(train): 1.335      loss(val): 1.980        time: 42.684s
[30]     loss(train): 1.305      loss(val): 1.963        time: 42.618s
[31]     loss(train): 1.279      loss(val): 1.968        time: 42.627s
[32]     loss(train): 1.256      loss(val): 1.968        time: 42.611s
[33]     loss(train): 1.231      loss(val): 1.970        time: 42.636s
[34]     loss(train): 1.208      loss(val): 1.983        time: 42.652s
[35]     loss(train): 1.183      loss(val): 1.993        time: 42.717s
[36]     loss(train): 1.160      loss(val): 2.003        time: 42.534s
[37]     loss(train): 1.142      loss(val): 1.992        time: 42.526s
[38]     loss(train): 1.120      loss(val): 1.998        time: 42.396s
[39]     loss(train): 1.101      loss(val): 1.984        time: 42.671s
[40]     loss(train): 1.083      loss(val): 1.983        time: 42.661s
[––– TRANSLATE (after train) ---]
[0] ein mann steht auf einem baugerüst . ->  A welder standing on a tiled surface . 
[1] männer spielen auf trommeln . ->  LeBron James plays drums while playing drums . 
[2] ein kind läuft auf einem weg . ->  A kid running away from a short ledge .  
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
      "factor": 0.95,
      "patience": 10
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