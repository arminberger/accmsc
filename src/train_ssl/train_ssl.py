import math


import self_supervised.data as data

import self_supervised.ss_training as ss_training
import torch
import utils.ml_utils as torch_utils
import os
import self_supervised.ss_paradigms as ss_paradigms
from self_supervised.loss import NTXentLoss
from self_supervised.loss import NTXentLossNew
from self_supervised import torch_datasets
import self_supervised.get_ss_models as get_ss_models
from constants import project_constants

def run_simclr_cap24_weighted_subject_wise(
    augs,
    paths_cfg,
    low_pass_freq=20,
    backbone_out_dim=1024,
    projection_out_dim=128,
    backbone_name="resnet_tiny",
    window_len=30,
    num_workers=0,
    pretrained_simclr_dict_path=None,
    num_epochs=100,
    weighted=True,
    batch_size=256,
    num_subjects=1,
    night_only=False,
    grad_checkpointing=False,
    linear_scaling=False,
    use_adam=True,
    weight_decay=True,
    autocast=False,
    normalize_data=False,
    provided_data=None,
):
    """

    Args:
        augs: List of augmentations to use, names of augmentations can be "na", "ap_p", "shuffle", "jit_scal","lfc","perm_jit","resample","noise","scale","negate","t_flip", "rotation", "perm", "t_warp", "hfc", "p_shift", "ap_f".
        low_pass_freq: The low pass frequency to use for the data
        backbone_out_dim: The output dimension of the backbone network
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

    Returns:

    """
    cap24_path = project_constants.GET_DATASET_PATH('cap24')
    DEVICE = torch_utils.get_available_device()
    print(f"Using device: {DEVICE}")
    cap24_path = cap24_path
    augs_str = [f"${aug}$" for aug in augs]
    augs_str = "_".join(augs_str)
    HUMAN_ACT_MAX_FREQ = low_pass_freq
    sampling_rate = 2 * HUMAN_ACT_MAX_FREQ
    WINDOW_LEN_SEC = window_len

    # Get the backbone network
    backbone_output_dim = backbone_out_dim
    backbone_name = backbone_name
    backbone_network = get_ss_models.get_model(
        name=backbone_name, output_dim=backbone_output_dim, grad_checkpointing=grad_checkpointing
    )
    # Get the SimCLR model
    projection_output_dim = projection_out_dim
    model = ss_paradigms.SimCLR(
        backbone=backbone_network,
        backbone_name=backbone_name,
        backbone_output_dim=backbone_output_dim,
        projector_output_dim=projection_output_dim,
    )

    # Load the pretrained SimCLR model
    if pretrained_simclr_dict_path is not None:
        model.load_state_dict(torch.load(pretrained_simclr_dict_path))
    # Print size of resnet in MB
    param_size = 0
    for param in backbone_network.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in backbone_network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Backbone model size: {size_all_mb}MB")
    # Print number of trainable parameters
    print(
        f"Number of trainable parameters in backbone: "
        f"{str(sum(p.numel() for p in backbone_network.parameters() if p.requires_grad))}"
    )
    # Print the model
    print(backbone_network)

    # Get the data
    split_ratio = 0.2
    if provided_data is None:
        (
            train_dict_list,
            val_dict_list,
            test_dict_list,
            train_weigths,
        ) = data.prep_cap24_lazy_aug_per_subject(
            augs=augs,
            dataset_path=cap24_path,
            SLIDING_WINDOW_LEN=sampling_rate * WINDOW_LEN_SEC,
            num_workers=num_workers,
            HUMAN_ACT_MAX_FREQ=HUMAN_ACT_MAX_FREQ,
            test_ratio=0,
            val_ratio=split_ratio,
            device=DEVICE,
            WINDOW_LEN_S=WINDOW_LEN_SEC,
            night_only=night_only,
            normalize_data=normalize_data,
        )
    else:
        train_dict_list, val_dict_list = provided_data
        test_dict_list = None
        train_weigths = None

    # Set up training parameters and dataloaders
    batch_size = batch_size
    learning_rate = 1e-3
    epochs = num_epochs
    dataloader_workers = 0


    criterion = NTXentLossNew(DEVICE, batch_size, temperature=0.1)
    criterion = criterion.to(DEVICE)
    # Use linear scaling for the optimizer
    k_factor = math.ceil(batch_size / 256)
    learning_rate = learning_rate if not linear_scaling else learning_rate * k_factor
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0 if not weight_decay else 1e-4) if use_adam else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=0
    )
    """
    # Use warmup scheduler
    def schedule(epoch):
        start_factor = 1 / k_factor
        lr = learning_rate
        warmup = 5
        if epoch < warmup:
            lr = epoch * (1/(start_factor*warmup)) * (learning_rate * start_factor)
        return k_factor * lr
    scheduler = None if not linear_scaling else torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
    model_name = (
        f"simclr_bn_{backbone_name}_augs_{augs_str}_lr_{learning_rate}"
        f"_bs_{batch_size}_outdim_{backbone_output_dim}_sr_{sampling_rate}"
        f"_winlen_{WINDOW_LEN_SEC}_projdim_{projection_output_dim}_wght_{weighted}_subj_{num_subjects}_no_{night_only}_ls_{linear_scaling}"
        f"_gc_{grad_checkpointing}_adam_{use_adam}_wd_{weight_decay}_ac_{autocast}_norm_{normalize_data}"
    )

    # Set up datasets and dataloaders

    # Create dataloaders
    train_sets = []
    for i in range(len(train_dict_list)):
        # Get augmented dataset
        aug_sets = list(train_dict_list[i].values())
        aug_set = torch_datasets.CombinedDataset(datasets=aug_sets)
        train_sets.append(aug_set)

    val_loaders = []
    for i in range(len(val_dict_list)):
        aug_sets=list(val_dict_list[i].values())
        aug_set = torch_datasets.CombinedDataset(datasets=aug_sets)
        val_loaders.append(
            torch.utils.data.DataLoader(
                aug_set,
                batch_size=batch_size,
                num_workers=dataloader_workers,
                pin_memory=True,
            )
        )






    # Train the model
    best_model = ss_training.train_simclr_precomputed_augs_per_subject(
        DEVICE=DEVICE,
        batch_size=batch_size,
        checkpoint_path=project_constants.SS_OUTPUT_MODELS_PATH,
        train_sets=train_sets,
        val_loaders=val_loaders,
        model=model,
        epochs=epochs,
        ntxent_criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        model_name=model_name,
        dataset_weights=train_weigths,
        num_subjects_per_set=num_subjects,
        grad_checkpointing=grad_checkpointing,
        autocast=autocast,
    )
    # Save the best model (best_model already is a state_dict)
    torch.save(best_model, os.path.join(project_constants.SS_OUTPUT_MODELS_PATH, f"best_model_{augs_str}.pt"))