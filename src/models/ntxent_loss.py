import torch
import torch.nn.functional as F

class NTXentLossNew(torch.nn.Module):
    def __init__(
        self,
        device,
        batch_size,
        temperature=0.1,
        norm_hidden=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = torch.tensor(temperature, dtype=torch.float32).to(device)
        self.large_num = torch.tensor(1e9, dtype=torch.int).to(device)
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.norm_hidden = norm_hidden
        self.mask = (
            F.one_hot(torch.arange(batch_size), num_classes=batch_size)
            .float()
            .to(device)
        )
        self.labels = torch.arange(batch_size).long().to(device)

    def forward(self, zis, zjs):
        loss = contrastive_loss(
            hidden1=zis,
            hidden2=zjs,
            device=self.device,
            hidden_norm=self.norm_hidden,
            temperature=self.temperature,
            mask=self.mask,
            labels=self.labels,
            large_num=self.large_num,
            batch_size=self.batch_size,
        )
        return loss

def contrastive_loss(
    hidden1,
    hidden2,
    device,
    temperature,
    large_num,
    batch_size,
    hidden_norm=True,
    weights=None,
    mask=None,
    labels=None,
):
    """Compute loss for model.

    Args:
        hidden1: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden2: hidden vector (`Tensor`) of shape (bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        weights: a weighting number or vector.

    Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
    """
    # Get (normalized) hidden1 and hidden2.
    if hidden_norm:
        hidden1 = F.normalize(hidden1, dim=-1, p=2)
        hidden2 = F.normalize(hidden2, dim=-1, p=2)
    batch_size = batch_size

    # Gather hidden1/hidden2 across replicas and create local labels.
    # Note: PyTorch does not have an exact equivalent to TPU cross-replica operations.
    # If you're using DistributedDataParallel, the tensors are already synchronized across GPUs.
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(batch_size).long().to(device) if labels is None else labels
    masks = (
        F.one_hot(torch.arange(batch_size), num_classes=batch_size).float().to(device)
        if mask is None
        else mask
    )

    LARGE_NUM = large_num
    logits_aa = torch.div(torch.mm(hidden1, hidden1_large.t()), temperature)
    logits_aa = torch.sub(logits_aa, torch.mul(masks, LARGE_NUM))
    logits_bb = torch.div(torch.mm(hidden2, hidden2_large.t()), temperature)
    logits_bb = torch.sub(logits_bb, torch.mul(masks, LARGE_NUM))
    logits_ab = torch.div(torch.mm(hidden1, hidden2_large.t()), temperature)
    logits_ba = torch.div(torch.mm(hidden2, hidden1_large.t()), temperature)

    loss_a = F.cross_entropy(
        input=torch.cat([logits_ab, logits_aa], dim=1),
        target=labels,
        reduction="sum",
        weight=weights,
    )
    loss_b = F.cross_entropy(
        input=torch.cat([logits_ba, logits_bb], dim=1),
        target=labels,
        reduction="sum",
        weight=weights,
    )
    loss = torch.add(loss_a, loss_b)

    return loss
