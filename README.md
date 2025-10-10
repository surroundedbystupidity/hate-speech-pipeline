# Hate Speech Detection

## Command Line Arguments

| Argument              | Type   | Default                    | Description                                           |
| --------------------- | ------ | -------------------------- | ----------------------------------------------------- |
| --mode                | str    | "diffusion"                | One of `"diffusion"` or `"classification"`.           |
| --generate-embeddings | flag   | False                      | Don't assume embeddings are available, generate them. |
| --train-file-path     | string | train_dataset_with_emb.csv | Path to the training CSV file.                        |
| --test-file-path      | string | test_dataset_with_emb.csv  | Path to the test CSV file.                            |
| --val-file-path       | string | val_dataset_with_emb.csv   | Path to the validation CSV file.                      |
| --subset-count        | int    | 500                        | Number of samples to use from the dataset.            |
| --window-size-hours   | int    | 1                          | Number of hours to use for snapshot window.           |
| --epochs              | int    | 10                         | Number of epochs to train for.                        |

## Sample Commands

### Diffusion

1. Look at a subset of records, **recommended to run this first**.

```
python main.py \
 --train-file-path retrain_train80.csv \
 --test-file-path retrain_test10.csv \
 --val-file-path retrain_validation10.csv \
 --subset-count=1000 \
 --generate-embeddings
```

1. Without embeddings, generate them.

```
python main.py \
 --train-file-path retrain_train80.csv \
 --test-file-path retrain_test10.csv \
 --val-file-path retrain_validation10.csv \
 --generate-embeddings
```

1. Files contain embeddings, no need to generate them. This is for subsequent runs, when files with embeddings are already saved.

```
python main.py \
 --train-file-path retrain_train80_with_embeddings.csv \
 --test-file-path retrain_test10_with_embeddings.csv \
 --val-file-path retrain_validation10_with_embeddings.csv \
```

### Node Classification

```
python main.py \
 --mode classification \
 --train-file-path retrain_train80_with_embeddings.csv \
 --test-file-path retrain_test10_with_embeddings.csv \
 --val-file-path retrain_validation10_with_embeddings.csv \
 --epochs 20
```


|    |   threshold |      mse |   accuracy |   precision |   recall |   f1_score |
|---:|------------:|---------:|-----------:|------------:|---------:|-----------:|
|  0 |         0.1 | 0.549267 |   0.450733 |  0.00858322 | 0.522222 |        nan |