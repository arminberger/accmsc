import hydra
from omegaconf import DictConfig
from src.data_download import get_dataset
from src.train_classifier import run_classification
from src.train_ssl import run_simclr_cap24_weighted_subject_wise
from src.utils import get_available_device
import os




@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    for path in cfg.paths:
        # Check if the path exists and create it if it doesn't
        if not os.path.exists(cfg.paths[path]):
            os.makedirs(cfg.paths[path])
            print(f"Created directory: {cfg.paths[path]}")

    if cfg.task == "download":
        datasets_to_download = cfg.download
        print(f'Downloading datasets: {[dataset.name for dataset in datasets_to_download.values()]}')
        for dataset in datasets_to_download:
            get_dataset(
                name=datasets_to_download[dataset].name,
                download_dir=cfg.paths.datasets,
                is_zip=datasets_to_download[dataset].is_zipped,
                url=datasets_to_download[dataset].url
            )
    elif cfg.task == "train_classifier":
        run_classification(
            feature_extractor_name=cfg.feature_extractor.network.name,
            classifier_name=cfg.classifier.network.name,
            dataset_cfg=cfg.classifier.dataset,
            device=get_available_device(),
            train_label_transform_dict=cfg.classifier.label_transform.transform,
            test_label_transform_dict=cfg.classifier.label_transform.transform,
            paths_cfg=cfg.paths,
            dataset_sampling_rate=cfg.classifier.dataset.sampling_rate,
            checkpoint_save_path=cfg.paths.model_checkpoints,
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
    elif cfg.task == "train_ssl":
        print("Training SSL")
        run_simclr_cap24_weighted_subject_wise(dataset_cfg=cfg.feature_extractor.dataset,
                                               augs=cfg.feature_extractor.augmentations,
                                               paths_cfg=cfg.paths,
                                               low_pass_freq=cfg.feature_extractor.network.low_pass_freq,
                                               sampling_rate=cfg.feature_extractor.network.input_sample_rate,
                                               backbone_name=cfg.feature_extractor.network.name,
                                               window_len=cfg.feature_extractor.network.input_len_seconds,
                                               num_epochs=60,
                                               batch_size=cfg.batch_size,
                                               num_subjects=4,
                                               grad_checkpointing=False,
                                               use_adam=True,
                                               weight_decay=True,
                                               autocast=True,
                                               normalize_data=cfg.normalize_data, )
    else:
        raise ValueError(f"Unknown task: {cfg.task}")

if __name__ == "__main__":
    # Set environment variables
    os.environ["WANDB_MODE"] = "offline"
    main()
