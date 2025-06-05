import math
import torch
from src.data_preprocessing import make_dataset
from src.utils import split_subject_wise
from src.torch_datasets import AccDataset, PrecomputedFeaturesDataset, ListDataset
from src.train_classifier import get_full_classification_model
import random
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import KFold
from src.train_classifier import train_model
import wandb
import time

def run_classification(
    feature_extractor_name,
    classifier_name,
    dataset_cfg,
    device,
    train_label_transform_dict,
    test_label_transform_dict,
    checkpoint_save_path,
    paths_cfg,
    model_params=None,
    feature_extractor_local_path=None,
    prev_window=0,
    post_window=0,
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
    looocv=True,
    weight_decay=1e-4,
    do_select_model=True,
    viterbi=False,
    batch_norm_after_feature_extractor=False,
):
    """

    Args:
        feature_extractor_name: One of the names specified in constants.model_constants.FEATURE_EXTRACTOR_NAMES
        classifier_name: One of the names specified in constants.model_constants.CLASSIFIER_NAMES
        dataset_name: One of the names specified in constants.dataset_constants.DATASET_NAMES
        device: Which device to run the model on
        model_params: Model params, depends on the feature extractor. Usually a dictionary with key 'augs'
        prev_window: How many previous windows to include in the input of an LSTM classifier
        post_window: How many subsequent windows to include in the input of an LSTM classifier
        train_test_split: The ratio of the dataset to use for testing
        weighted_sampling: Whether to use weighted sampling or not
        label_transform_name: How to transform the labels. One of the names specified in constants.dataset_constants.DATASET_LABELS_TRANSFORMS
        test_label_transform_name: unused
        human_act_freq_cutoff: What frequency to use for the low pass filter. If None it is derived from the feature extractor sampling frequency
        freeze_foundational_model: Whether to fine tune the feature extractor or not
        precompute_features: Whether to precompute the features or not (makes the training a lot faster, but can't fine tune the feature extractor)
        num_epochs: For how many epochs to train
        seed: Random seed
        normalize_data: Whether to normalize the data or not
        classifier_drouput: Dropout for the LSTM classifier
        cross_validation: How many folds to use for cross validation, if 0 then no cross validation is used
        looocv: Whether to use leave one out cross validation or not. If True then cross_validation must be 0
        weight_decay: What weight decay to use for the optimizer
        do_select_model: Whether to select the best model based on the validation loss or not
        viterbi: Whether to use viterbi post processing or not
        batch_norm_after_feature_extractor: Whether to use batch normalization as described in the thesis or not

    Returns:

    """


    dataset_name = dataset_cfg.name
    # Add all the parameters to the tensorboard writer

    wandb_run = wandb.init(project='Classifier Training', config={
        'feature_extractor_name': feature_extractor_name,
        'classifier_name': classifier_name,
        'dataset_name': dataset_name,
        'model_params': model_params,
        'prev_window': prev_window,
        'post_window': post_window,
        'train_test_split': train_test_split,
        'weighted_sampling': weighted_sampling,
        'human_act_freq_cutoff': human_act_freq_cutoff,
        'freeze_foundational_model': freeze_foundational_model,
        'precompute_features': precompute_features,
        'num_epochs': num_epochs,
        'seed': seed,
        'normalize_data': normalize_data,
        'classifier_drouput': classifier_drouput,
        'cross_validation': cross_validation,
        'looocv': looocv,
        'weight_decay': weight_decay,
        'do_select_model': do_select_model,
        'viterbi': viterbi,
        'batch_norm_after_feature_extractor': batch_norm_after_feature_extractor,
        'Current Time and Date': time.strftime("%m/%d/%Y %H:%M:%S", time.localtime()),
    })

    torch.manual_seed(seed)

    if not freeze_foundational_model and precompute_features:
        raise ValueError(
            "Cannot precompute features if foundational model is not frozen."
        )
    # check if dataset_name is list or not
    if not isinstance(dataset_name, list):
        dataset_name = [dataset_name]
    # Compute number of classes of the transform
    label_transform = train_label_transform_dict
    vals = list(label_transform.values())
    vals, _ = np.unique(vals, return_counts=True)
    num_classes = len(vals)
    (
        my_model,
        sampling_rate,
        input_len_sec,
        feature_extractor_output_len,
        feature_extractor,
        feature_extractor_filename,
    ) = get_full_classification_model(
        feature_extractor_name=feature_extractor_name,
        feature_extractor_local_path=feature_extractor_local_path,
        model_params=model_params,
        classifier_name=classifier_name,
        device=device,
        num_classes=num_classes,
        freeze_foundational_model=freeze_foundational_model,
        assemble_feature_extractor=not precompute_features,
        prev_window=prev_window,
        post_window=post_window,
        return_feature_ext_filename=True,
        dropout=classifier_drouput,
        batch_norm_after_feature_extractor=batch_norm_after_feature_extractor,
    )
    summary_string += (
        f"feature_extractor_filename: \t \t \t {feature_extractor_filename} \n"
    )
    if human_act_freq_cutoff is None:
        human_act_freq_cutoff = math.floor(sampling_rate / 2)

    # Load data: First preprocess the data and get list of per-subject dataframes, then
    dataset_list = []
    subject_ids = []
    for name in dataset_name:
        (
            dataset_list_name,
            labels_list_name,
            subject_ids_name,
        ) = make_dataset(
            dataset_cfg=dataset_cfg,
            paths_cfg=paths_cfg,
            target_sampling_rate=sampling_rate,
            low_pass_filter_freq=human_act_freq_cutoff,
            try_cached=True,
            win_len_s=input_len_sec,
            normalize_data=normalize_data,
        )
        dataset_list_name = [
            AccDataset(
                motion_df=dataset_list_name[k],
                label_df=labels_list_name[k],
                num_samples=sampling_rate * input_len_sec,
                label_transform=train_label_transform_dict,
            )
            for k in range(len(dataset_list_name))
        ]

        dataset_list.extend(dataset_list_name)
        subject_ids.extend(subject_ids_name)

    if precompute_features:
        dataset_list = [
            PrecomputedFeaturesDataset(
                feature_extractor=feature_extractor,
                feature_extractor_output_length=feature_extractor_output_len,
                acc_dataset=dataset_list[i],
                device=device,
            )
            for i in range(len(dataset_list))
        ]

    checkpoint_save_name = (
        f"base_{str(feature_extractor_name)}_freeze_{str(freeze_foundational_model)}_precompute_features_"
        f"{str(precompute_features)}_classifier_{classifier_name}"
        f"_dataset_{str(dataset_name)}_weighted_sampling_{weighted_sampling}_prewin_{prev_window}_postwin_{post_window}"
        f"_normalize_data_{normalize_data}"
    )
    try:
        aug_add = f'_augs_{str(model_params["augs"])}'
        checkpoint_save_name += aug_add
    except:
        pass


    if looocv:
        f1s = []
        balaccs = []
        kappas = []
        # Leave one out cross validation
        for i in range(len(dataset_list)):
            # Get new model for each fold
            (
                my_model,
                sampling_rate,
                input_len_sec,
                feature_extractor_output_len,
                feature_extractor,
                feature_extractor_filename,
            ) = (
                get_full_classification_model(
                feature_extractor_name=feature_extractor_name,
                feature_extractor_local_path=feature_extractor_local_path,
                model_params=model_params,
                classifier_name=classifier_name,
                device=device,
                num_classes=num_classes,
                freeze_foundational_model=freeze_foundational_model,
                assemble_feature_extractor=not precompute_features,
                prev_window=prev_window,
                post_window=post_window,
                return_feature_ext_filename=True,
                dropout=classifier_drouput,
                batch_norm_after_feature_extractor=batch_norm_after_feature_extractor,
            ))
            # Split data into train_list and test_list
            train_list = dataset_list[:i] + dataset_list[i + 1 :]
            test_list = [dataset_list[i]]
            # Further split train_list into train and val
            train_list, val_list = split_subject_wise(
                train_list,
                test_ratio=train_test_split if do_select_model else 0,
                random_gen=random.Random(seed + i),
            )
            train = ListDataset(
                dataset_list=train_list,
                prev_elements=prev_window,
                post_elements=post_window,
            )

            val_dataloader = [
                DataLoader(
                    ListDataset(
                        dataset_list=[x],
                        prev_elements=prev_window,
                        post_elements=post_window,
                    ),
                    batch_size=512,
                    shuffle=False,
                )
                for x in val_list
            ]
            test_dataloader = [
                DataLoader(
                    ListDataset(
                        dataset_list=[x],
                        prev_elements=prev_window,
                        post_elements=post_window,
                    ),
                    batch_size=512,
                    shuffle=False,
                )
                for x in test_list
            ]
            sampler = None
            if weighted_sampling:
                sampler = get_weighted_sampler(train)
            train_dataloader = DataLoader(
                train,
                batch_size=64,
                shuffle=True if sampler is None else False,
                sampler=sampler,
            )

            # Train model
            f1, kappa, balacc = train_model(
                my_model=my_model,
                train_dataloader=train_dataloader,
                train_list=[ListDataset([x], prev_elements=prev_window, post_elements=post_window) for x in train_list],
                test_dataloaders=test_dataloader,
                val_dataloaders=val_dataloader,
                checkpoint_save_name=checkpoint_save_name,
                checkpoint_save_path=checkpoint_save_path,
                num_epochs=num_epochs,
                wandb_run=wandb_run,
                device=device,
                num_fold=i,
                weight_decay=weight_decay,
                do_selection=do_select_model,
                labels_transform_dict=test_label_transform_dict,
                viterbi=viterbi,
            )
            f1s.append(f1)
            kappas.append(kappa)
            balaccs.append(balacc)
            # Print current results
            print(
                f"LOOCV current results, Fold {i}: \n f1: {np.mean(f1s)} +/- {np.std(f1s)} \n kappa: {np.mean(kappas)} +/- {np.std(kappas)} \n balacc: {np.mean(balaccs)} +/- {np.std(balaccs)}"
            )
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s)
        kappa_mean = np.mean(kappas)
        kappa_std = np.std(kappas)
        balacc_mean = np.mean(balaccs)
        balacc_std = np.std(balaccs)
        wandb_run.log({
            "LOOOCV f1 mean": f1_mean,
            "LOOOCV f1 std": f1_std,
            "LOOOCV kappa mean": kappa_mean,
            "LOOOCV kappa std": kappa_std,
            "LOOOCV balacc mean": balacc_mean,
            "LOOOCV balacc std": balacc_std,
        })

    elif cross_validation > 0:

        kf = KFold(n_splits=cross_validation, shuffle=True, random_state=seed)
        # Aggregate
        f1s = []
        kappas = []
        balaccs = []
        reports = []
        for k, (train_indices, test_indices) in enumerate(kf.split(dataset_list)):

            # Get new model for each fold
            (
                my_model,
                sampling_rate,
                input_len_sec,
                feature_extractor_output_len,
                feature_extractor,
                feature_extractor_filename,
            ) = get_full_classification_model(
                feature_extractor_name=feature_extractor_name,
                feature_extractor_local_path=feature_extractor_local_path,
                model_params=model_params,
                classifier_name=classifier_name,
                device=device,
                num_classes=num_classes,
                freeze_foundational_model=freeze_foundational_model,
                assemble_feature_extractor=not precompute_features,
                prev_window=prev_window,
                post_window=post_window,
                return_feature_ext_filename=True,
                dropout=classifier_drouput,
                batch_norm_after_feature_extractor=batch_norm_after_feature_extractor,
            )

            print(f"Cross validation fold {k+1}")
            checkpoint_save_name_k = checkpoint_save_name + f"_fold_{k+1}"
            train_list_k = [dataset_list[i] for i in train_indices]
            test_list_k = [dataset_list[i] for i in test_indices]
            # Split into train and val
            train_list_k, val_list_k = split_subject_wise(
                train_list_k,
                test_ratio=train_test_split if do_select_model else 0,
                random_gen=random.Random(seed + k),
            )
            train = ListDataset(
                dataset_list=train_list_k,
                prev_elements=prev_window,
                post_elements=post_window,
            )
            sampler = None
            if weighted_sampling:
                sampler = get_weighted_sampler(train)

            train_dataloader = DataLoader(
                train,
                batch_size=64,
                shuffle=(True if sampler is None else False),
                sampler=sampler,
            )

            test_dataloader = [
                DataLoader(
                    ListDataset(
                        dataset_list=[x],
                        prev_elements=prev_window,
                        post_elements=post_window,
                    ),
                    batch_size=512,
                    shuffle=False,
                )
                for x in test_list_k
            ]
            val_dataloader = (
                [
                    DataLoader(
                        ListDataset(
                            dataset_list=[x],
                            prev_elements=prev_window,
                            post_elements=post_window,
                        ),
                        batch_size=512,
                        shuffle=False,
                    )
                    for x in val_list_k
                ]
                if do_select_model
                else None
            )

            f1, kappa, balacc, report = train_model(
                my_model=my_model,
                train_dataloader=train_dataloader,
                train_list=[ListDataset([x], prev_elements=prev_window, post_elements=post_window) for x
                            in train_list_k],
                test_dataloaders=test_dataloader,
                val_dataloaders=val_dataloader,
                checkpoint_save_name=checkpoint_save_name_k,
                checkpoint_save_path=checkpoint_save_path,
                num_epochs=num_epochs,
                wandb_run=wandb_run,
                device=device,
                num_fold=k,
                weight_decay=weight_decay,
                do_selection=do_select_model,
                labels_transform_dict=test_label_transform_dict,
                viterbi=viterbi,
                return_report=True,
            )
            f1s.append(f1)
            kappas.append(kappa)
            balaccs.append(balacc)
            reports.append(report)
            # Print current results
            print(
                f"Current results, Fold {k}: \n f1: {np.mean(f1s)} +/- {np.std(f1s)} \n kappa: {np.mean(kappas)} +/- {np.std(kappas)} \n balacc: {np.mean(balaccs)} +/- {np.std(balaccs)}"
            )
        f1_mean = np.mean(f1s)
        f1_std = np.std(f1s)
        kappa_mean = np.mean(kappas)
        kappa_std = np.std(kappas)
        balacc_mean = np.mean(balaccs)
        balacc_std = np.std(balaccs)
        cross_validation_result = f"Cross validation results: \n f1: {f1_mean} +/- {f1_std} \n kappa: {kappa_mean} +/- {kappa_std} \n balacc: {balacc_mean} +/- {balacc_std}"
        print(cross_validation_result)
        print(reports)
        wandb_run.log({
            "Cross validation f1 mean": f1_mean,
            "Cross validation f1 std": f1_std,
            "Cross validation kappa mean": kappa_mean,
            "Cross validation kappa std": kappa_std,
            "Cross validation balacc mean": balacc_mean,
            "Cross validation balacc std": balacc_std,
        })

    else:
        # Split data into train_list and test_list
        train_list, test_list = split_subject_wise(
            dataset_list, test_ratio=train_test_split, random_gen=random.Random(seed)
        )

        test_dataloader = [DataLoader(ListDataset(
            dataset_list=[x], prev_elements=prev_window, post_elements=post_window
        ), batch_size=512, shuffle=False) for x in test_list]
        print("Train without cross validation")
        train_list, val_list = split_subject_wise(
            train_list, test_ratio=train_test_split, random_gen=random.Random(seed)
        )
        print(f"Train size (subjects): {len(train_list)}")
        train = ListDataset(
            dataset_list=train_list, prev_elements=prev_window, post_elements=post_window
        )
        sampler = None
        if weighted_sampling:
            sampler = get_weighted_sampler(train)

        train_dataloader = DataLoader(
            train,
            batch_size=64,
            shuffle=(True if sampler is None else False),
            sampler=sampler,
        )

        val_dataloader = [DataLoader(ListDataset(
            dataset_list=[x], prev_elements=prev_window, post_elements=post_window
        ), batch_size=512, shuffle=False) for x in val_list]
        train_model(
            my_model=my_model,
            train_dataloader=train_dataloader,
            train_list=[ListDataset([x], prev_elements=prev_window, post_elements=post_window) for x in
                        train_list],
            test_dataloaders=test_dataloader,
            val_dataloaders=val_dataloader,
            checkpoint_save_name=checkpoint_save_name,
            checkpoint_save_path=checkpoint_save_path,
            num_epochs=num_epochs,
            wandb_run=wandb_run,
            device=device,
            num_fold=0,
            weight_decay=weight_decay,
            labels_transform_dict=test_label_transform_dict,
        )

    wandb_run.finish(0)


def get_weighted_sampler(train):
    train_indices_ws = list(range(len(train)))
    train_labels_ws = [train[i][1] for i in train_indices_ws]
    class_labels_count = np.array(
        [
            len(np.asarray(train_labels_ws == t).nonzero()[0])
            for t in np.unique(train_labels_ws)
        ]
    )
    class_weights = 1.0 / class_labels_count
    samples_weight = np.array([class_weights[t] for t in train_labels_ws])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        samples_weight.type("torch.DoubleTensor"),
        len(samples_weight),
        generator=torch.Generator(),
    )
    print(sampler.generator.initial_seed())
    return sampler