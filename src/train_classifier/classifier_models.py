import os
import re
import torch
from src.models import Resnet, RNNModel, LSTMModel, SimCLR
from src.train_ssl import get_backbone_network
from torch import nn
import zipfile

def get_full_classification_model(
    feature_extractor_name,
    classifier_cfg,
    num_classes,
    device,
    feature_extractor_local_path=None,
    model_params=None,
    freeze_foundational_model=False,
    assemble_feature_extractor=True,
    return_feature_ext_filename=False,
    dropout=0,
    batch_norm_after_feature_extractor=False,
):
    """

    Args:
        feature_extractor_name:
        classifier_name:
        num_classes:
        window_size:
        freeze_foundational_model:
        assemble_feature_extractor:

    Returns:

    """
    prev_window = classifier_cfg.prev_windows
    post_window = classifier_cfg.post_windows
    classifier_name = classifier_cfg.name
    clip_gradients = classifier_cfg.clip_gradients
    (
        feature_extractor,
        sampling_rate,
        input_len_sec,
        output_len,
        feature_extractor_filename,
    ) = get_feature_extractor(
        name=feature_extractor_name,
        local_path=feature_extractor_local_path,
        freeze=freeze_foundational_model,
        model_params=model_params,
        return_filename=return_feature_ext_filename,
        batch_norm_after_feature_extractor=batch_norm_after_feature_extractor,
    )
    if assemble_feature_extractor:
        my_model = get_classification_model(
            name=classifier_name,
            feature_ext_output_size=output_len,
            num_classes=num_classes,
            prev_window=prev_window,
            post_window=post_window,
            feature_extractor=feature_extractor,
            device=device,
            dropout=dropout,
            clip_gradients=clip_gradients,
        )
    else:
        my_model = get_classification_model(
            name=classifier_name,
            feature_ext_output_size=output_len,
            prev_window=prev_window,
            post_window=post_window,
            num_classes=num_classes,
            feature_extractor=None,
            device=device,
            dropout=dropout,
            clip_gradients=clip_gradients,
        )

    print(my_model)
    if return_feature_ext_filename:
        return (
            my_model,
            sampling_rate,
            input_len_sec,
            output_len,
            feature_extractor,
            feature_extractor_filename,
        )
    else:
        return (
            my_model,
            sampling_rate,
            input_len_sec,
            output_len,
            feature_extractor,
        )


def get_classification_model(
    name,
    feature_ext_output_size,
    num_classes,
    device,
    prev_window=0,
    post_window=0,
    feature_extractor=None,
    dropout=0,
    clip_gradients=False,
):
    """

    Args:
        name:
        feature_ext_output_size:
        num_classes:
        window_size:
        feature_extractor: If None, the model will take as inputs outputs from the feature extractor

    Returns:

    """
    if name == 'naive_mlp':
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feature_ext_output_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_classes),
        )
        if feature_extractor is not None:
            model = torch.nn.Sequential(feature_extractor, model)
    elif name == 'naive_mlp_small':
        model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(feature_ext_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes),
        )
        if feature_extractor is not None:
            model = torch.nn.Sequential(feature_extractor, model)
    elif name == 'rnn_classifier':
        # RNN Classifier
        model = RNNModel(
            input_size=feature_ext_output_size,
            hidden_dim=256,
            num_classes=num_classes,
            device=device,
            feature_extractor=feature_extractor,
            use_all_hidden=True,
            sequence_length=prev_window + post_window + 1,
        )
    elif name == 'lstm_classifier':
        # LSTM Classifier
        model = LSTMModel(
            input_size=feature_ext_output_size,
            hidden_dim=128,
            num_classes=num_classes,
            device=device,
            feature_extractor=feature_extractor,
            use_all_hidden=False,
            bidireactional=False,
            sequence_length=prev_window + post_window + 1,
            dropout=dropout,
            num_layers=2,
        )
    elif name == 'asleep_lstm':
        model = LSTMModel(
            input_size=feature_ext_output_size,
            hidden_dim=128,
            num_classes=num_classes,
            device=device,
            feature_extractor=feature_extractor,
            use_all_hidden=False,
            bidireactional=True,
            sequence_length=prev_window + post_window + 1,
            dropout=dropout,
            num_layers=2,
        )
    else:
        raise ValueError(
            f"Classifier name {name} not recognized."
        )
    if clip_gradients:
        # Clip gradients to avoid exploding gradients
        for p in model.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
    return model


def get_feature_extractor(
    name,
    local_path,
    freeze=False,
    model_params=None,
    return_filename=False,
    batch_norm_after_feature_extractor=False,
):
    """

    :param name:
    :param local_path:
    :param repo: torch.hub repo. If both loacl_path and repo are specified, local_path is used
    :param model_params: model parameters for the feature extractor (see relevant if statement for details)
    :return: Returns the feature extractor model, sampling rate, input length in seconds, and output length
    """
    filename_model = "FILENAME EXTARCTION NOT IMPLEMENTED YET"

    if 'harnet30' == name:
        sampling_rate, input_len_sec, output_len = 30, 30, 1024
        zip_path = os.path.join(local_path, 'harnet30_ukb.zip')
        # unzip the zip file to a folder with the same name
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(local_path)
        local_path = os.path.join(local_path, 'harnet30_ukb')

        harnet30 = torch.hub.load(
            local_path, "harnet30", class_num=5, pretrained=True, source="local"
        )
        model = list(harnet30.children())[0]
    elif "harnet10_cap24" == name:
        local_path = os.path.join(local_path, 'harnet10_cap24.mdl')
        sampling_rate, input_len_sec, output_len = 30, 10, 1024

        model = Resnet(
            output_size=2, is_eva=True, resnet_version=1, epoch_len=input_len_sec
        )
        del model.classifier.linear1
        del model.classifier.linear2
        print(model)
        rm_keys = [
            "aot_h.linear1.weight",
            "aot_h.linear1.bias",
            "scale_h.linear1.weight",
            "scale_h.linear1.bias",
            "permute_h.linear1.weight",
            "permute_h.linear1.bias",
            "time_w_h.linear1.weight",
            "time_w_h.linear1.bias",
        ]
        # Load CAP24 .mdl from local path
        sd = torch.load(local_path, map_location=torch.device("cpu"))
        print(sd.keys())
        for key in rm_keys:
            del sd[key]
        model.load_state_dict(sd)
        print(model)
        model = list(model.children())[0]
    elif "harnet10_ukb" == name:
        local_path = os.path.join(local_path, 'harnet10_ukb.mdl')
        sampling_rate, input_len_sec, output_len = 30, 10, 1024

        model = Resnet(
            output_size=2, is_eva=True, resnet_version=1, epoch_len=input_len_sec
        )
        del model.classifier.linear1
        del model.classifier.linear2
        print(model)
        rm_keys = [
            "aot_h.linear1.weight",
            "aot_h.linear1.bias",
            "scale_h.linear1.weight",
            "scale_h.linear1.bias",
            "permute_h.linear1.weight",
            "permute_h.linear1.bias",
            "time_w_h.linear1.weight",
            "time_w_h.linear1.bias",
        ]
        # Load CAP24 .mdl from local path
        sd = torch.load(local_path, map_location=torch.device("cpu"))
        sd_keys = list(sd.keys())
        for key in sd_keys:
            # Only remove keys that start with 'module.'
            prefix = "module."
            print(key)
            if key.startswith(prefix):
                sd[key[len(prefix) :]] = sd[key]
                del sd[key]
        for key in rm_keys:
            del sd[key]
        model.load_state_dict(sd)
        print(model)
        model = list(model.children())[0]
    elif "harnet10_untrained" == name:
        # Load harnet trained by me on custom data and augs
        augs = model_params["augs"]
        # local path is only base path in this case
        model_path = None
        for path in os.scandir(local_path):
            print(path.name)
            if path.name.endswith(".pt") and path.name.startswith('best_model'):
                # Remove best_model_ prefix
                filename = path.name.replace('best_model_', '')
                # Extract the characters of filename from the start to the first $ sign
                filename = filename.split('$')[0]
                # Check that filename is resnet_harnet
                if filename.startswith('harnet10_untrained'):
                    if augs is not None:
                        print(set(augs))
                        print(set(extract_augs(path.name)))
                        if set(augs) == set(extract_augs(path.name)):
                            model_path = path.path
                            filename_model = path.name
                            break
                    else:
                        # Just take the first model found
                        model_path = path.path
                        filename_model = path.name
        if model_path is None:
            raise ValueError("No model found with the specified augmentations")
        sampling_rate, input_len_sec, output_len = 30, 10, 1024
        # Get the backbone network
        backbone_output_dim = 1024
        backbone_name = "harnet10_untrained"
        projection_output_dim = 128
        model = load_own_simclr(
            backbone_name, model_path, backbone_output_dim, projection_output_dim
        )
    elif "plain_resnet" == name:
        """
        If model_params is not None, it should be an integer with the input length of the motion data in seconds
        If model_params is None, the default input length is 30 seconds
        """
        if model_params is None:
            sampling_rate, input_len_sec, output_len = 30, 30, 1024
        else:
            sampling_rate, input_len_sec, output_len = 30, model_params, 1024
        model = Resnet(
            output_size=2, is_eva=True, resnet_version=1, epoch_len=input_len_sec
        )
        model = list(model.children())[0]
        print(model)
    elif 'my_resnet_cap24' == name:
        # Load my resnet trained on cap24
        sampling_rate, input_len_sec, output_len = 20, 30, 1024
        # Get the backbone network
        backbone_output_dim = 1024
        backbone_name = "resnet_large"
        projection_output_dim = 128
        model = load_own_simclr(
            backbone_name, local_path, backbone_output_dim, projection_output_dim
        )

    elif 'my_resnet_new' == name:
        sampling_rate, input_len_sec, output_len = 40, 30, 1024
        backbone_output_dim = 1024
        backbone_name = "resnet_small"
        projection_output_dim = 128
        model = load_own_simclr(
            backbone_name, local_path, backbone_output_dim, projection_output_dim
        )
    elif 'resnet_tiny' == name:
        sampling_rate, input_len_sec, output_len = 40, 10, 1024
        backbone_output_dim = 1024
        backbone_name = "resnet_tiny"
        projection_output_dim = 128

        augs = model_params["augs"]

        # local path is only base path in this case
        model_path = None
        for path in os.scandir(local_path):
            print(path.name)
            if path.name.endswith(".pt"):
                if augs is not None:
                    print(set(augs))
                    print(set(extract_augs(path.name)))
                    if set(augs) == set(extract_augs(path.name)):
                        model_path = path.path
                        filename_model = path.name
                        break
                else:
                    # Just take the first model found
                    model_path = path.path
                    filename_model = path.name
        if model_path is None:
            raise ValueError("No model found with the specified augmentations")
        model = load_own_simclr(
            backbone_name, model_path, backbone_output_dim, projection_output_dim
        )
        # Move .pt file to CONSUMED_MODELS_PATH
        # shutil.move(model_path, os.path.join(project_constants.CONSUMED_MODELS_PATH, filename_model))
    elif 'resnet_mid' == name:
        sampling_rate, input_len_sec, output_len = 40, 10, 1024
        backbone_output_dim = 1024
        backbone_name = "resnet_mid"
        projection_output_dim = 128
        augs = model_params["augs"]

        # local path is only base path in this case
        model_path = None
        for path in os.scandir(local_path):
            print(path.name)
            if path.name.endswith(".pt"):
                print(set(augs))
                print(set(extract_augs(path.name)))
                if set(augs) == set(extract_augs(path.name)):
                    model_path = path.path
                    filename_model = path.name
                    break
        if model_path is None:
            raise ValueError("No model found with the specified augmentations")
        model = load_own_simclr(
            backbone_name, model_path, backbone_output_dim, projection_output_dim
        )

    else:
        raise ValueError(
            "Invalid feature extractor name. Got: "
            + str(name)
        )
    model = nn.Sequential(
        model,
        nn.Flatten(),
        nn.BatchNorm1d(output_len, track_running_stats=(False if freeze else True))
        if batch_norm_after_feature_extractor
        else nn.Identity(),
    )

    if freeze:
        model.requires_grad_(False)
        model.eval()
    else:
        model.requires_grad_(True)
    if return_filename:
        return model, sampling_rate, input_len_sec, output_len, filename_model
    else:
        return model, sampling_rate, input_len_sec, output_len


def extract_augs(file_name):
    # Regular expression to match the pattern of augmentations
    pattern = re.compile(r"\$[^$]+\$")

    # Find all matches in the file name
    matches = pattern.findall(file_name)

    # Remove the dollar sign wrapper and return the list of augmentations
    augs = [match.replace("$", "") for match in matches]

    return augs


def load_own_simclr(backbone_name, local_path, output_dim, proj_dim):
    backbone = get_backbone_network(backbone_name, output_dim)
    model = SimCLR(
        backbone=backbone,
        backbone_output_dim=output_dim,
        projector_output_dim=proj_dim,
    )
    loaded = torch.load(local_path, map_location=torch.device("cpu"))
    model.load_state_dict(
        torch.load(local_path, map_location=torch.device("cpu"))
    )
    model = model.backbone
    return model



