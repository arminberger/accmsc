import math
import random
import os
import torch
import numpy as np
import wandb
from src.utils import get_available_device
from src.train_ssl import get_backbone_network, train_simclr_precomputed_augs_per_subject
from src.models import SimCLR, NTXentLossNew
from src.data_preprocessing import make_dataset
from src.utils import split_subject_wise
from src.torch_datasets import SSLazyAugDatasetSubjectWise, SSRawDataset, CombinedDataset


def run_simclr_cap24_weighted_subject_wise(dataset_cfg, augs, paths_cfg, low_pass_freq=15, sampling_rate=30,
                                           feature_vector_dim=1024, projection_out_dim=128, backbone_name="resnet_tiny",
                                           window_len=30, pretrained_simclr_dict_path=None,
                                           num_epochs=100, batch_size=256, num_subjects=1,
                                           night_only=False, grad_checkpointing=False, linear_scaling=False,
                                           use_adam=True, weight_decay=True, autocast=False, normalize_data=False,
                                           provided_data=None, train_val_split_ratio=0.2,
                                           wandb_project="SSL-Training", wandb_run_name=None):
    """

    Args:
        augs: List of augmentations to use, names of augmentations can be "na", "ap_p", "shuffle", "jit_scal","lfc","perm_jit","resample","noise","scale","negate","t_flip", "rotation", "perm", "t_warp", "hfc", "p_shift", "ap_f".
        low_pass_freq: The low pass frequency to use for the data
        feature_vector_dim: The output dimension of the backbone network
        projection_out_dim: The output dimension of the projectior in the SimCLR terminology
        backbone_name: The name of the backbone network to use, names can be "resnet_tiny", "resnet_small", "resnet_mid", and definitions can be found in the source/self_supervised/get_ss_models.py file.
        window_len: The length of the window to use for the self supervised training, in seconds.
        num_workers: The number of workers to use for data preprocessing
        pretrained_simclr_dict_path: The path to the pretrained SimCLR model, if None, then the model will be trained from scratch.
        num_epochs: The number of epochs to train the model for.
        weighted: Whether to use weighted sampling according to standard deviation of the data or not.
        batch_size: The batch size to use for training.
        num_subjects: The number of subject to sample a batch from.
        night_only: Whether to use only night data or not.
        grad_checkpointing: Whether to use gradient checkpointing or not.
        linear_scaling: Whether to use linear scaling of the learning rate or not.
        use_adam: Whether to use Adam or SGD as the optimizer.
        weight_decay: Whether to use weight decay or not.
        autocast: Whether to use autocast (mixed precision training) or not.
        normalize_data: Whether to normalize the data or not.
        provided_data: Specialized argument used for efficient training of many models on the same data. More in the source/main_ss_augeval_final.py file.
        wandb_project: Name of the wandb project to log to.
        wandb_run_name: Custom name for the wandb run. If None, a name will be generated automatically.

    Returns:

    """
    # Initialize wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config={
        'augs': augs,
        'low_pass_freq': low_pass_freq,
        'sampling_rate': sampling_rate,
        'feature_vector_dim': feature_vector_dim,
        'projection_out_dim': projection_out_dim,
        'backbone_name': backbone_name,
        'window_len': window_len,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'num_subjects': num_subjects,
        'night_only': night_only,
        'grad_checkpointing': grad_checkpointing,
        'linear_scaling': linear_scaling,
        'use_adam': use_adam,
        'weight_decay': weight_decay,
        'autocast': autocast,
        'normalize_data': normalize_data
    })
    DEVICE = get_available_device()
    print(f"Using device: {DEVICE}")

    # Check if sampling rate is at least twice the low pass frequency
    if sampling_rate < 2 * low_pass_freq:
        raise ValueError(
            f"Sampling rate {sampling_rate} is too low for the low pass frequency {low_pass_freq}"
        )

    # Get the backbone network
    backbone_network = get_backbone_network(
        name=backbone_name, output_dim=feature_vector_dim, grad_checkpointing=grad_checkpointing
    )

    # Get the SimCLR model
    model = SimCLR(
        backbone=backbone_network,
        backbone_output_dim=feature_vector_dim,
        projector_output_dim=projection_out_dim,
    )

    # Load the pretrained SimCLR model
    if pretrained_simclr_dict_path is not None:
        model.load_state_dict(torch.load(pretrained_simclr_dict_path))

    print_backbone_info(backbone_network)

    # Get the data
    print('Loading data...')
    if provided_data is None:
        motion_data_list, subject_ids = make_dataset(
            dataset_cfg,
            paths_cfg,
            target_sampling_rate=sampling_rate,
            low_pass_filter_freq=low_pass_freq,
            normalize_data=normalize_data,
            win_len_s=window_len,
        )
        dataset_per_subject_list = []
        for subject in motion_data_list:
            dataset_per_subject_list.append(
                SSRawDataset(
                    motion_df_list=[subject],
                    win_length_samples=window_len * sampling_rate,
                    channels_first=False,
                )
            )
        train, val = split_subject_wise(
            dataset_per_subject_list, train_val_split_ratio, random_gen=random.Random(42)
        )

        # Compute weighted sampler weights – weights are not connected to aug
        train_weights = compute_weights(train)
        # Dataset dict should have a list of subjects for every augmentation
        train_dict_list = compute_aug_dataset_list(augs, train)
        val_dict_list = compute_aug_dataset_list(augs, val)

    else:
        train_dict_list, val_dict_list = provided_data
        train_weights = None

    # Set up training parameters and dataloaders
    print('Setting up training...')
    batch_size = batch_size
    learning_rate = 1e-3
    epochs = num_epochs
    dataloader_workers = 0

    criterion = NTXentLossNew(DEVICE, batch_size, temperature=0.1)
    criterion = criterion.to(DEVICE)
    # Use linear scaling for the optimizer
    k_factor = math.ceil(batch_size / 256)
    learning_rate = learning_rate if not linear_scaling else learning_rate * k_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0 if not weight_decay else 1e-4) if use_adam else torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # Use warmup scheduler
    def schedule(epoch):
        start_factor = 1 / k_factor
        lr = learning_rate
        warmup = 5
        if epoch < warmup:
            lr = epoch * (1 / (start_factor * warmup)) * (learning_rate * start_factor)
        return k_factor * lr

    scheduler = None if not linear_scaling else torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)

    augs_str = [f"${aug}$" for aug in augs]
    augs_str = "_".join(augs_str)
    model_name = (
        f"simclr_bn_{backbone_name}_augs_{augs_str}_lr_{learning_rate}"
        f"_bs_{batch_size}_outdim_{feature_vector_dim}_sr_{sampling_rate}"
        f"_winlen_{window_len}_projdim_{projection_out_dim}_subj_{num_subjects}_no_{night_only}_ls_{linear_scaling}"
        f"_gc_{grad_checkpointing}_adam_{use_adam}_wd_{weight_decay}_ac_{autocast}_norm_{normalize_data}"
    )

    # Set up datasets and dataloaders

    # Create dataloaders
    train_sets = []
    for i in range(len(train_dict_list)):
        # Get augmented dataset
        aug_sets = list(train_dict_list[i].values())
        aug_set = CombinedDataset(datasets=aug_sets)
        train_sets.append(aug_set)

    val_loaders = []
    for i in range(len(val_dict_list)):
        aug_sets = list(val_dict_list[i].values())
        aug_set = CombinedDataset(datasets=aug_sets)
        val_loaders.append(
            torch.utils.data.DataLoader(
                aug_set,
                batch_size=batch_size,
                num_workers=dataloader_workers,
                pin_memory=True,
            )
        )

    # Train the model
    best_model = train_simclr_precomputed_augs_per_subject(
        DEVICE=DEVICE,
        batch_size=batch_size,
        checkpoint_path=paths_cfg.model_checkpoints,
        train_sets=train_sets,
        val_loaders=val_loaders,
        model=model,
        epochs=epochs,
        ntxent_criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=model_name,
        dataset_weights=train_weights,
        num_subjects_per_set=num_subjects,
        grad_checkpointing=grad_checkpointing,
        autocast=autocast,
    )
    # Save the best model (best_model already is a state_dict)
    torch.save(best_model, os.path.join(paths_cfg.model_checkpoints, f"best_model_{augs_str}.pt"))


def print_backbone_info(backbone_network):
    # Print size of resnet in MB
    param_size = 0
    for param in backbone_network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in backbone_network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"Backbone model size: {size_all_mb}MB")
    # Print number of trainable parameters
    print(
        f"Number of trainable parameters in backbone: "
        f"{str(sum(p.numel() for p in backbone_network.parameters() if p.requires_grad))}"
    )
    # Print the model
    print(backbone_network)


def compute_weights(dataset_per_subject_list):
    def compute_st_deviation(data):
        assert data.shape[1] == 3
        assert len(data.shape) == 2
        std = np.std(data, axis=0)
        assert std.shape[0] == 3
        return std.sum() / 3

    weight_per_subject = []
    for i in range(len(dataset_per_subject_list)):
        raw = dataset_per_subject_list[i]
        weights = torch.zeros(len(raw))
        for j in range(len(raw)):
            # raw[i] will be a vector of size (#channels, SLIDING_WINDOW_LEN)
            weights[j] = float(compute_st_deviation(raw[j].numpy(force=True)))
        weight_per_subject.append(weights)
    return weight_per_subject


def compute_aug_dataset_list(augs, datasets):
    aug_datasets = []
    for i in range(len(datasets)):
        aug_datasets.append(compute_aug_dataset(augs, datasets[i]))
    return aug_datasets


def compute_aug_dataset(augs, dataset):
    dict = {}
    for aug in augs:
        aug_train_dataset = SSLazyAugDatasetSubjectWise(
            dataset, aug=aug
        )

        dict[aug] = aug_train_dataset
    return dict
