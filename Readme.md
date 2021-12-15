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
```
2021-12-15 14:55:32,895 [INFO] -- LOAD DATA
2021-12-15 14:55:38,549 [INFO] -- LOAD TRANSFORMER
2021-12-15 14:55:42,696 [INFO] -- TRANSLATE
2021-12-15 14:55:42,775 [INFO] -- [0] ein mann steht auf einem baugerüst . ->  busking busking busking busking busking busking busking busking busking busking busking busking busking
2021-12-15 14:55:42,836 [INFO] -- [1] männer spielen auf trommeln . ->  busking busking busking busking busking busking busking busking busking busking busking
2021-12-15 14:55:42,907 [INFO] -- [2] ein kind läuft auf einem weg . ->  busking busking busking busking busking busking busking busking busking busking busking busking busking
2021-12-15 14:55:42,907 [INFO] -- BEGIN TRAINING
2021-12-15 14:58:54,110 [INFO] -- [5]	loss(train): 3.713	loss(val): 3.645	time: 38.062s
2021-12-15 15:02:09,529 [INFO] -- [10]	loss(train): 3.035	loss(val): 3.163	time: 38.027s
2021-12-15 15:05:21,580 [INFO] -- [15]	loss(train): 2.588	loss(val): 2.836	time: 36.221s
2021-12-15 15:08:39,135 [INFO] -- [20]	loss(train): 2.169	loss(val): 2.515	time: 40.039s
2021-12-15 15:11:52,358 [INFO] -- [25]	loss(train): 1.794	loss(val): 2.251	time: 38.010s
2021-12-15 15:15:04,557 [INFO] -- [30]	loss(train): 1.482	loss(val): 2.103	time: 36.341s
2021-12-15 15:18:21,331 [INFO] -- [35]	loss(train): 1.256	loss(val): 2.028	time: 38.034s
2021-12-15 15:21:26,276 [INFO] -- [40]	loss(train): 1.069	loss(val): 2.022	time: 35.944s
2021-12-15 15:24:29,340 [INFO] -- [45]	loss(train): 0.890	loss(val): 1.961	time: 35.925s
2021-12-15 15:27:35,764 [INFO] -- [50]	loss(train): 0.824	loss(val): 1.966	time: 36.456s
2021-12-15 15:30:40,611 [INFO] -- [55]	loss(train): 0.780	loss(val): 1.963	time: 36.468s
2021-12-15 15:33:46,107 [INFO] -- [60]	loss(train): 0.767	loss(val): 1.964	time: 36.701s
2021-12-15 15:36:52,160 [INFO] -- [65]	loss(train): 0.764	loss(val): 1.966	time: 36.639s
2021-12-15 15:39:57,619 [INFO] -- [70]	loss(train): 0.761	loss(val): 1.966	time: 36.572s
2021-12-15 15:43:02,891 [INFO] -- [75]	loss(train): 0.758	loss(val): 1.967	time: 36.480s
2021-12-15 15:46:08,083 [INFO] -- [80]	loss(train): 0.755	loss(val): 1.968	time: 36.442s
2021-12-15 15:49:13,058 [INFO] -- [85]	loss(train): 0.754	loss(val): 1.968	time: 36.499s
2021-12-15 15:52:17,821 [INFO] -- [90]	loss(train): 0.754	loss(val): 1.969	time: 36.376s
2021-12-15 15:55:22,424 [INFO] -- [95]	loss(train): 0.751	loss(val): 1.971	time: 36.420s
2021-12-15 15:58:26,937 [INFO] -- [100]	loss(train): 0.750	loss(val): 1.971	time: 36.346s
2021-12-15 16:01:32,177 [INFO] -- [105]	loss(train): 0.744	loss(val): 1.972	time: 36.696s
2021-12-15 16:04:37,684 [INFO] -- [110]	loss(train): 0.744	loss(val): 1.972	time: 36.499s
2021-12-15 16:07:43,065 [INFO] -- [115]	loss(train): 0.743	loss(val): 1.974	time: 36.422s
2021-12-15 16:10:47,750 [INFO] -- [120]	loss(train): 0.741	loss(val): 1.974	time: 36.386s
2021-12-15 16:11:52,465 [INFO] -- User interrupted training, trying to proceed and save model
2021-12-15 16:11:53,616 [INFO] -- Loaded best model from checkpoint
2021-12-15 16:11:53,616 [INFO] -- [47]	loss(train): 0.856	loss(val): 1.960
2021-12-15 16:11:53,616 [INFO] -- TRANSLATE
2021-12-15 16:11:53,681 [INFO] -- [0] ein mann steht auf einem baugerüst . ->  An action car is standing on a sports court . 
2021-12-15 16:11:53,744 [INFO] -- [1] männer spielen auf trommeln . ->  The musicians are playing the drums on the drums . 
2021-12-15 16:11:53,796 [INFO] -- [2] ein kind läuft auf einem weg . ->  A kid runs away from a cliff . 

```
![Loss plot](results/01-sinusodial-position/loss.png)

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
    "batch_size": 32,
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