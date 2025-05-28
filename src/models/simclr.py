from torch import nn

class SimCLR(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_output_dim,
        projector_output_dim=128,
    ):
        """_summary_

        Args:
            backbone (_type_): _description_
            backbone_name (_type_): _description_
            sample_input (_type_):
            projector_output_dim (int, optional): _description_. Defaults to 128.
        """
        super(SimCLR, self).__init__()

        self.backbone = backbone
        self.backbone_output_dim = backbone_output_dim
        # Projector has to be dropped when using the backbone to train downstream tasks
        self.projector = Projector(
            model="SimCLR",
            input_dim=self.backbone_output_dim,
            inter_dim=self.backbone_output_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, x1, x2):

        z1 = self.backbone(x1)
        z2 = self.backbone(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)

        z1 = self.projector(z1)
        z2 = self.projector(z2)

        return z1, z2


class Projector(nn.Module):
    def __init__(self, model, input_dim, inter_dim, output_dim):
        super(Projector, self).__init__()
        if model == "SimCLR":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, inter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(inter_dim, output_dim),
                nn.Flatten(),
            )
        elif model == "byol":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, inter_dim, bias=False),
                nn.BatchNorm1d(inter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(inter_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim, affine=False),
            )
        elif model == "NNCLR":
            self.projector = nn.Sequential(
                nn.Linear(input_dim, inter_dim, bias=False),
                nn.BatchNorm1d(inter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(inter_dim, inter_dim, bias=False),
                nn.BatchNorm1d(inter_dim),
                nn.ReLU(inplace=True),
                nn.Linear(inter_dim, output_dim, bias=False),
                nn.BatchNorm1d(output_dim),
            )
        elif model == "TS-TCC":
            self.projector = nn.Sequential(
                nn.Linear(output_dim, input_dim // 2),
                nn.BatchNorm1d(input_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim // 2, input_dim // 4),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x
