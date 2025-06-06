To replicate the locked environment install uv and run the following command:

```bash
uv sync
```

This will create a virtual environment and install all the dependencies in it.

Before running the training and evaluation, you need to download the datasets. This might take some time.

```bash
uv run python main.py task=download
```

To run on Euler use the following command:

```bash
sbatch --job-name=uv-main --time=24:00:00 --gpus-per-node=1 --cpus-per-task=1 --mem-per-cpu=32G --output=uv-main-%j.out --wrap="uv run python main.py"
```

To train with e.g. different augmentation settings, pass this command as the argument to --wrap:

```bash
uv run python main.py feature_extractor/augmentations=best_bsc
```

To specify which augmentations to look for during classification, one can use the files specified in `feature_extractor/augmentations`:

```bash
uv run python main.py feature_extractor/augmentations@classifier.backbone_augs=harnet
```

To train a classifier using a backbone that was trained using this projects self-supervised learning, copy the `best_model_...` file (e.g. `best_model_resnet_harnet_$rotation$_$lfc$_$t_warp_harnet$_hash_2621969038.pt`) to the trained_checkpoints directory.
