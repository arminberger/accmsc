import torch
import os
from src.train_ssl import gen_aug
import copy
from tqdm import tqdm
import random

def model_loss_augs_precomputed(
    framework, aug_sample1, aug_sample2, model, criterion, DEVICE, grad_checkpointing=False, autocast=False
):
    '''

    Args:
        framework:
        aug_sample1: Assumed to have a batch size of an integer power of 2 and >64
        aug_sample2: Assumed to have a batch size of an integer power of 2 and >64
        model:
        criterion:
        DEVICE:

    Returns:

    '''

    aug_sample1, aug_sample2 = (
        aug_sample1.type(torch.float32).to(DEVICE),
        aug_sample2.type(torch.float32).to(DEVICE),
    )

    if framework == "simclr":
        # Check if CUDA is available
        if torch.cuda.is_available() and autocast:
            with torch.cuda.amp.autocast():
                loss = criterion(model(x1=aug_sample1, x2=aug_sample2))
        else:
            loss = criterion(model(x1=aug_sample1, x2=aug_sample2))
    return loss

def model_loss(framework, sample, aug1, aug2, model, criterion, DEVICE):
    aug_sample1 = gen_aug(sample, aug1)
    aug_sample2 = gen_aug(sample, aug2)
    aug_sample1, aug_sample2 = (
        aug_sample1.type(torch.float32).to(DEVICE),
        aug_sample2.type(torch.float32).to(DEVICE),
    )
    if framework == "simclr":
        z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        loss = criterion(z1, z2)
    return loss



def train_simclr_precomputed_augs_per_subject(
    train_sets,
    val_loaders,
    model,
    epochs,
    DEVICE,
    batch_size,
    checkpoint_path,
    ntxent_criterion,
    optimizer,
    scheduler,
    model_name,
    dataset_weights=None,
    num_subjects_per_set=1,
    grad_checkpointing=False,
    autocast=False
):
    """

    Args:
        train_loader: DataLoader for training data, each sample is a tuple of augmentations,
        two of which are selected at random for each batch
        val_loader: DataLoader for validation data, each sample is a tuple of augmentations
        model: SimCLR model
        epochs:
        DEVICE:
        batch_size:
        checkpoint_path:
        ntxent_criterion:
        optimizer:
        scheduler:
        model_name: Name of the model to save checkpoints as

    Returns:

    """

    # Set up local variables
    best_model = None
    min_val_loss = 1e8
    seed = 42
    model.to(DEVICE)
    ntxent_criterion.to(DEVICE)


    # Main training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        # Train epoch
        total_train_loss = 0
        random.Random(seed + epoch).shuffle(train_sets)
        if dataset_weights is not None:
            random.Random(seed + epoch).shuffle(dataset_weights)
        for i in tqdm(range(len(train_sets) // num_subjects_per_set)):
            dataset = torch.utils.data.ConcatDataset(
                train_sets[i * num_subjects_per_set : (i + 1) * num_subjects_per_set]
            )
            # concat weigths
            weigths_concat = None
            train_sampler = None
            if dataset_weights is not None:

                weights = dataset_weights[
                    i * num_subjects_per_set : (i + 1) * num_subjects_per_set
                ]
                for weight in weights:
                    if weigths_concat is None:
                        weigths_concat = weight
                    else:
                        weigths_concat = torch.cat((weigths_concat, weight))
                train_sampler = torch.utils.data.WeightedRandomSampler(
                    weigths_concat,
                    len(weigths_concat),
                    generator=torch.Generator().manual_seed(2147483647),
                )
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True if weigths_concat is None else False,
                sampler=train_sampler,
                num_workers=0,
                drop_last=True,
                pin_memory=True,
            )
            total_train_loss = total_train_loss + train_epoch(
                DEVICE,
                batch_size,
                model,
                ntxent_criterion,
                optimizer,
                seed,
                train_loader,
                silent=True,
                grad_checkpointing=grad_checkpointing,
                autocast=autocast
            )

        print(f"Total Training Loss of Epoch {epoch}: {total_train_loss}")


        # Compute model_name
        save_name = f"{model_name}_epoch_{str(epoch)}.pt"
        # save model
        model_dir = os.path.join(checkpoint_path, save_name)
        print("Saving model at epoch {} to {}".format(epoch, model_dir))
        torch.save({"model_state_dict": model.state_dict()}, model_dir)

        # Compute validation error
        curr_val_loss = 0
        for val_loader in val_loaders:
            curr_val_loss = curr_val_loss + compute_validation_precomputed_augs(
                DEVICE,
                batch_size,
                ntxent_criterion,
                model,
                val_loader,
                silent=True,
            )

        # TODO: Maybe add warmup
        if scheduler is not None:
            scheduler.step()
        # Save best model (with the lowest validation loss)
        print(f"Total Validation Loss of Epoch {epoch}: {curr_val_loss}")

        if curr_val_loss <= min_val_loss:
            min_val_loss = curr_val_loss
            best_model = copy.deepcopy(model.state_dict())
            print(f"Saving best model at epoch {epoch}")
    return best_model


def train_epoch(
    DEVICE,
    batch_size,
    model,
    ntxent_criterion,
    optimizer,
    seed,
    train_loader,
    silent=False,
    grad_checkpointing=False,
    autocast=False,
):
    total_loss = torch.tensor(0.0, device=DEVICE)
    model.train()
    progress = tqdm(train_loader) if not silent else train_loader

    for idx, samples in enumerate(progress):
        seed = seed + 1
        sample1_index, sample2_index = random.Random(seed).sample(
            range(len(samples)), k=2
        )
        sample1 = samples[sample1_index]
        sample2 = samples[sample2_index]

        optimizer.zero_grad()
        if sample1.size(0) != batch_size:
            continue
        loss = model_loss_augs_precomputed(
            framework="simclr",
            aug_sample1=sample1,
            aug_sample2=sample2,
            model=model,
            criterion=ntxent_criterion,
            DEVICE=DEVICE,
            grad_checkpointing=grad_checkpointing,
            autocast=autocast
        )



        loss.backward()
        optimizer.step()
        total_loss = torch.add(total_loss, loss)

    return total_loss.item()


def compute_validation_precomputed_augs(
    DEVICE, batch_size, criterion, model, val_loader, silent=False
):
    # Compute validation error
    with torch.no_grad():
        model.eval()
        total_loss = torch.tensor(0.0, device=DEVICE)
        n_batches = 0
        if not silent:
            print("Computing validation loss")
        progress = tqdm(val_loader) if not silent else val_loader
        for idx, samples in enumerate(progress):
            for sample1_index in range(len(samples)):
                for sample2_index in range(len(samples)):
                    if sample1_index < sample2_index:

                        sample1 = samples[sample1_index]
                        sample2 = samples[sample2_index]
                        # Samples have to have requires grad true for gradient checkpointing to not throw warning - should not affect anything
                        sample1.requires_grad = True
                        sample2.requires_grad = True
                        if sample1.size(0) != batch_size:
                            continue
                        n_batches += 1
                        loss = model_loss_augs_precomputed(
                            framework="simclr",
                            aug_sample1=sample1,
                            aug_sample2=sample2,
                            model=model,
                            criterion=criterion,
                            DEVICE=DEVICE,
                        )
                        total_loss = torch.add(total_loss, loss)
    return total_loss.item()



