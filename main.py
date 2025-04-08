import hydra
from omegaconf import DictConfig
from src.data_download import get_dataset

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.task == "download":
        get_dataset(
            name=cfg.dataset.name,
            download_dir=cfg.datasets_dir,
            is_zip=cfg.dataset.is_zipped,
            url=cfg.dataset.url
        )

if __name__ == "__main__":
    main()
    