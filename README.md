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