import hydra
from omegaconf import DictConfig
from src.data_download import get_dataset
from src.train_classifier import run_classification
from src.utils import get_available_device

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if cfg.task == "download":
        get_dataset(
            name=cfg.classification_dataset.name,
            download_dir=cfg.datasets_dir,
            is_zip=cfg.classification_dataset.is_zipped,
            url=cfg.classification_dataset.url
        )
    if cfg.task == "train_classifier":
        run_classification(
            feature_extractor_name=cfg.feature_extractor.network.name,
            classifier_name=cfg.classifier.network.name,
            dataset_name=cfg.classifier.dataset.name,
            device=get_available_device(),
            train_label_transform_dict=cfg.classifier.label_transform.transform,
            test_label_transform_dict=cfg.classifier.label_transform.transform,
            dataset_sampling_rate=cfg.classifier.dataset.sampling_rate,
            checkpoint_save_path=cfg.model_checkpoint_dir,
            model_params={'augs': ['t_warp', 'lfc', 'rotation']},
            prev_window=8 if cfg.classifier.network.name == 'lstm_classifier' else 0,
            post_window=8 if cfg.classifier.network.name == 'lstm_classifier' else 0,
            train_test_split=0.2,
            weighted_sampling=False,
            human_act_freq_cutoff=None,
            freeze_foundational_model=True,
            precompute_features=True,
            num_epochs=100,
            seed=46012094831,
            normalize_data=False,
            classifier_drouput=0,
            cross_validation=0,
            looocv=False,
            weight_decay=1e-4,
            do_select_model=True,
            viterbi=False,
            batch_norm_after_feature_extractor=False,
        )


if __name__ == "__main__":
    main()
    