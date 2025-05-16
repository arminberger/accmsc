from src.models import ResNet1D
from src.models import Resnet
import torch

def get_backbone_network(name, output_dim, grad_checkpointing=False):
    if name == "resnet_large":
        return ResNet1D(
            in_channels=3,
            base_filters=64,
            kernel_size=5,
            stride=1,
            groups=1,
            n_block=9,
            increasefilter_gap=2,
            n_classes=output_dim,
            gradient_checkpointing=grad_checkpointing,
        )
    if name == "resnet_mid":
        return ResNet1D(
            in_channels=3,
            base_filters=64,
            kernel_size=5,
            stride=1,
            groups=1,
            n_block=8,
            increasefilter_gap=2,
            n_classes=output_dim,
            gradient_checkpointing=grad_checkpointing,
        )
    if name == "resnet_small":
        return ResNet1D(
            in_channels=3,
            base_filters=64,
            kernel_size=5,
            stride=1,
            groups=1,
            n_block=7,
            increasefilter_gap=2,
            n_classes=output_dim,
            gradient_checkpointing=grad_checkpointing,
        )
    if name == "resnet_tiny":
        return ResNet1D(
            in_channels=3,
            base_filters=64,
            kernel_size=5,
            stride=1,
            groups=1,
            n_block=4,
            increasefilter_gap=2,
            n_classes=output_dim,
            gradient_checkpointing=grad_checkpointing,
        )
    if name == "resnet_harnet":
        # Expects input with size (batch_size, 3, 300)
        model = Resnet(
            output_size=2, is_eva=True, resnet_version=1, epoch_len=10
        )
        model = model.feature_extractor
        # Reshape to remove last dimension of output
        model = torch.nn.Sequential(
            model,
            torch.nn.Flatten(),
        )
        return model
    else:
        raise ValueError(f"Unknown backbone name: {name}")
