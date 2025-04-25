from src.models import ResNet1D as resnet1d

def get_model(name, output_dim, grad_checkpointing=False):
    if name == "resnet_large":
        return resnet1d.ResNet1D(
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
        return resnet1d.ResNet1D(
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
        return resnet1d.ResNet1D(
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
        return resnet1d.ResNet1D(
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
