To replicate the locked environment install uv and run the following command:

```bash
uv sync
```

This will create a virtual environment and install all the dependencies in it.

Before running the training and evaluation, you need to download the datasets. This might take some time.

```bash
uv run python main.py task=download
```

