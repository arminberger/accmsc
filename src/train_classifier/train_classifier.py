import numpy as np
import torch
import os
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from src.utils import classification_train_loop as train_loop, classification_test_loop as test_loop
from torchmetrics.functional.classification import multiclass_cohen_kappa
from torchmetrics.functional.classification import multiclass_confusion_matrix
from torchmetrics.functional.classification import multiclass_f1_score
from torchmetrics.functional.classification import multiclass_accuracy
from hmmlearn.hmm import CategoricalHMM
import itertools
import copy


def train_model(
    my_model,
    train_dataloader,
    train_list,
    test_dataloaders,
    val_dataloaders,
    checkpoint_save_name,
    checkpoint_save_path,
    wandb_run,
    device,
    labels_transform_dict,
    num_epochs=100,
    timestamp_model_path=None,
    num_fold=0,
    weight_decay=1e-4,
    do_selection=True,
    viterbi=False,
    return_report=False,
):
    my_model.to(device)
    sample = train_dataloader.dataset[0][0].unsqueeze(0).to(device)
    my_model.eval()
    with torch.no_grad():
        num_classes = my_model(sample).shape[1]
    my_model.train()
    learning_rate = 1e-3
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        my_model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    if timestamp_model_path is not None:
        print("Loading model...")
        my_model.load_state_dict(torch.load(timestamp_model_path))
    """best_loss_model = copy.deepcopy(my_model.state_dict())
    best_f1_model = copy.deepcopy(my_model.state_dict())"""
    best_kappa_model = copy.deepcopy(my_model.state_dict())

    t = 0
    best_loss = int(1e9)
    best_loss_epoch = 20
    best_balanced_f1 = -1000
    best_balanced_f1_epoch = 20
    best_kappa = -1000
    best_kappa_epoch = 0
    hmm = None
    get_obs = None
    while t < num_epochs:
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss = train_loop(train_dataloader, my_model, device, loss_fn, optimizer)

        print(f"[WANDB LOG] Training Loss at step {t}")
        wandb_run.log({"Training Loss": train_loss})

        """loss, acc, balanced_acc = test_loop(
            val_dataloader, my_model, device, loss_fn, num_classes
        )"""
        if do_selection:
            # Test the model
            # Compute loss, balanced f1 and kappa
            preds, targets, preds_logits = compute_preds(
                my_model,
                device,
                val_dataloaders,
                return_unreduced=True,
                dataloader_per_subject=True,
            )

            """val_loss = loss_fn(preds_logits, targets)
            val_balanced_f1 = multiclass_f1_score(
                preds, targets, num_classes=num_classes
            )"""
            val_kappa = multiclass_cohen_kappa(preds, targets, num_classes=num_classes)
            print(f"Validation Kappa: {val_kappa}")

            """if val_loss < best_loss and t > best_loss_epoch:
                best_loss = val_loss
                best_loss_epoch = t
                best_loss_model = copy.deepcopy(my_model.state_dict())
            if val_balanced_f1 > best_balanced_f1 and t > best_balanced_f1_epoch:
                best_balanced_f1 = val_balanced_f1
                best_balanced_f1_epoch = t
                best_f1_model = copy.deepcopy(my_model.state_dict())"""
            if val_kappa > best_kappa and t > best_kappa_epoch:
                best_kappa = val_kappa
                best_kappa_epoch = t
                best_kappa_model = copy.deepcopy(my_model.state_dict())

            """writer.add_scalar(f"Validation Loss (Fold {num_fold})", val_loss, t)
            writer.add_scalar(
                f"Validation Balanced F1 (Fold {num_fold})", val_balanced_f1, t
            )"""
            # writer.add_scalar(f"Validation Kappa (Fold {num_fold})", val_kappa, t)
            print(f"[WANDB LOG] Validation Kappa (Fold {num_fold}) at step {t}")
            wandb_run.log({f"Validation Kappa (Fold {num_fold})": val_kappa}, step=t)

        if t % 10 == 0:
            filename = f"{checkpoint_save_name}_epoch_{str(t)}.pth"
            save_path = os.path.join(checkpoint_save_path, filename)
            print(f"Saving model to: {save_path}")

        # Test per class accuracy of my_model on test set
        """hmm, get_obs, invert = (
            getHMM(val_dataloaders, my_model, device, num_classes, n_buckets=3, obs='top_k_probs', top_k=1,
                   top_k_probs=1)
            if viterbi
            else (None, None, None)
        )
        preds, targets = compute_preds(
            my_model,
            device,
            test_dataloaders,
            dataloader_per_subject=True,
            viterbi=hmm,
            convert_sequence=get_obs,
        )
        kappa = multiclass_cohen_kappa(preds, targets, num_classes=num_classes)
        f1_score = multiclass_f1_score(preds, targets, num_classes=num_classes)
        acc = multiclass_accuracy(preds, targets, num_classes=num_classes)
        print(f"Test Kappa: {kappa}, Test Balanced F1: {f1_score}, Balanced Accuracy: {acc}")
        preds, targets = compute_preds(
            my_model,
            device,
            test_dataloaders,
            dataloader_per_subject=True,
            viterbi=None,
        )
        kappa = multiclass_cohen_kappa(preds, targets, num_classes=num_classes)
        f1_score = multiclass_f1_score(preds, targets, num_classes=num_classes)
        acc = multiclass_accuracy(preds, targets, num_classes=num_classes)
        print(f"Test Kappa w/o Viterbi: {kappa}, Test Balanced F1: {f1_score}, Balanced Accuracy: {acc}")
        writer.add_scalar(f"Test Kappa (Fold {num_fold})", kappa, t)
        writer.add_scalar(f"Test Balanced F1 (Fold {num_fold})", f1_score, t)"""

        t = t + 1

    if not do_selection:
        best_kappa_model = copy.deepcopy(my_model.state_dict())
    # Performance of best models on testset

    if do_selection:
        """# Best loss model
        my_model.load_state_dict(best_loss_model)

        report, _, _, _ = compute_report(
            device,
            loss_fn,
            my_model,
            num_classes,
            test_dataloaders,
            best_loss_epoch,
            dataloader_per_subject=True,
        )
        writer.add_text("Best Loss Model Report", report, best_loss_epoch)

        # Best f1 model
        my_model.load_state_dict(best_f1_model)
        report, _, _, _ = compute_report(
            device,
            loss_fn,
            my_model,
            num_classes,
            test_dataloaders,
            best_balanced_f1_epoch,
            dataloader_per_subject=True,
        )
        writer.add_text("Best F1 Model Report", report, best_balanced_f1_epoch)"""

    # Best kappa model
    my_model.load_state_dict(best_kappa_model)
    hmm, get_obs, invert = (
        getHMM(train_list, val_dataloaders, my_model, device, num_classes, n_buckets=3, obs='top_k_probs', top_k=2, top_k_probs=1)
        if viterbi
        else (None, None, None)
    )
    report, f1, kappa, balacc = compute_report(
        device,
        loss_fn,
        my_model,
        num_classes,
        test_dataloaders,
        best_kappa_epoch,
        labels_transform=labels_transform_dict,
        viterbi=hmm,
        convert_sequence=get_obs,
        dataloader_per_subject=True,
    )
    print(f"[WANDB LOG] Best Kappa Model Report (after all epochs)")
    wandb_run.log({"Best Kappa Model Report": report})

    print(report)

    print("Done!")

    return (f1, kappa, balacc) if not return_report else (f1, kappa, balacc, report)


def compute_report(
    device,
    loss_fn,
    my_model,
    num_classes,
    test_dataloader,
    epoch,
    labels_transform=None,
    viterbi=None,
    convert_sequence=None,
    dataloader_per_subject=False,
):
    preds, targets, preds_logits = compute_preds(
        my_model,
        device,
        test_dataloader,
        return_unreduced=True,
        dataloader_per_subject=dataloader_per_subject,
        viterbi=viterbi,
        convert_sequence=convert_sequence,
    )

    if labels_transform is not None:
        # Use numpy vectorize to apply the transform to all elements of the array
        num_classes_transformed = len(set(labels_transform.values()))
        transform_lambda = lambda x: labels_transform[x]
        targets = targets.apply_(transform_lambda)
        preds = preds.apply_(transform_lambda)

    val_loss = loss_fn(preds_logits, targets)
    val_balanced_f1 = multiclass_f1_score(
        preds,
        targets,
        num_classes=num_classes
        if labels_transform is None
        else num_classes_transformed,
    )
    val_kappa = multiclass_cohen_kappa(
        preds,
        targets,
        num_classes=num_classes
        if labels_transform is None
        else num_classes_transformed,
    )
    balanced_acc = multiclass_accuracy(
        preds,
        targets,
        num_classes if labels_transform is None else num_classes_transformed,
        average="macro",
    )
    acc = multiclass_accuracy(
        preds,
        targets,
        num_classes if labels_transform is None else num_classes_transformed,
        average="weighted",
    )
    confusion_matrix = multiclass_confusion_matrix(
        preds,
        targets,
        num_classes=num_classes
        if labels_transform is None
        else num_classes_transformed,
        normalize="true",
    )
    report = classification_report(targets, preds, digits=4)

    final_str = (
        f"Report \n"
        f"Model from epoch {epoch} \n"
        f"Loss: {val_loss} \n"
        f"Balanced F1: {val_balanced_f1} \n"
        f"Kappa: {val_kappa} \n"
        f"Balanced Accuracy: {balanced_acc} \n"
        f"Accuracy: {acc} \n"
        f"Confusion Matrix: {confusion_matrix} \n"
        f"Classification Report:\n {report}"
    )

    return final_str, val_balanced_f1, val_kappa, balanced_acc


def compute_metric(
    my_model, device, test_dataloader, metric_function, preds_first=True
):
    with torch.no_grad():
        preds = []
        targets = []
        for X, y in test_dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = my_model(X)
            preds.extend(pred.argmax(1).type(torch.int).tolist())
            targets.extend(y.type(torch.int).tolist())
        preds = torch.tensor(preds)
        targets = torch.tensor(targets)
        if preds_first:
            return metric_function(preds, targets)
        else:
            return metric_function(targets, preds)


def compute_preds(
    my_model,
    device,
    test_dataloaders,
    return_unreduced=False,
    viterbi=None,
    convert_sequence=None,
    dataloader_per_subject=False,
):
    if not dataloader_per_subject:
        test_dataloaders = [test_dataloaders]
    preds_full = None
    preds_reduced_full = None
    targets_full = None
    my_model.eval()
    for test_dataloader in test_dataloaders:
        with torch.no_grad():
            preds_reduced = None
            preds = None
            targets = []
            for X, y in test_dataloader:
                X = X.to(device)
                pred = my_model(X)

                if preds is None:
                    preds = pred.numpy(force=True)
                else:
                    preds = np.concatenate((preds, pred.numpy(force=True)), axis=0)
                if preds_reduced is None:
                    preds_reduced = pred.argmax(1).type(torch.int).tolist()
                else:
                    preds_reduced.extend(pred.argmax(1).type(torch.int).tolist())

                targets.extend(y.type(torch.int).tolist())

            preds = torch.tensor(preds)
            if viterbi is not None:
                if convert_sequence is None:
                    raise ValueError(
                        "convert_sequence and invert_sequence must be provided if viterbi is not None"
                    )
                # Use Viterbi algorithm to smooth predictions
                observations = convert_sequence(preds)

                _, best_seq = viterbi.decode(
                    observations.reshape(-1, 1), algorithm="viterbi"
                )
                preds_reduced = best_seq

            preds_reduced = torch.tensor(preds_reduced)
            targets = torch.tensor(targets)
        if preds_full is None:
            preds_full = preds
            preds_reduced_full = preds_reduced
            targets_full = targets
        else:
            preds_full = torch.cat((preds_full, preds), dim=0)
            preds_reduced_full = torch.cat((preds_reduced_full, preds_reduced), dim=0)
            targets_full = torch.cat((targets_full, targets), dim=0)

    if return_unreduced:
        return preds_reduced_full, targets_full, preds_full
    else:
        return preds_reduced_full, targets_full


def getHMM(train_list, val_loaders, model, device, num_classes, train=False, n_buckets=3, obs=None, top_k=3, top_k_probs=1):
    train_list = [DataLoader(x, batch_size=1024, shuffle=False) for x in train_list]
    train_list = [
        compute_preds(
            model, device, x, return_unreduced=True, dataloader_per_subject=False
        )
        for x in train_list
    ]
    preds_train = [x[2] for x in train_list]
    labels_train = [x[1].numpy(force=True) for x in train_list]
    val_loaders = [
        compute_preds(
            model, device, x, return_unreduced=True, dataloader_per_subject=False
        )
        for x in val_loaders
    ]
    preds = [x[2] for x in val_loaders]
    labels = [x[1].numpy(force=True) for x in val_loaders]


    num_hidden_states = num_classes

    if obs=="buckets":
        combinations = list(itertools.product(range(n_buckets), repeat=num_classes))
        combinations = list(itertools.product(range(num_classes), combinations))
        num_obs_states = len(combinations)
    elif obs=="top_k":
        if top_k > num_classes:
            raise ValueError("top_k must be smaller than or equal to num_classes")
        combinations = list(itertools.permutations(range(num_classes), top_k))
        num_obs_states = len(combinations)
    elif obs=="top_k_probs":
        if top_k_probs > num_classes or top_k > num_classes:
            raise ValueError("top_k and top_k_probs must be smaller than or equal to num_classes")
        combinations_top_k = list(itertools.permutations(range(num_classes), top_k))
        combinations_probs = list(itertools.product(range(n_buckets), repeat=top_k_probs))
        combinations = list(itertools.product(combinations_probs, combinations_top_k))
        num_obs_states = len(combinations)
    else:
        num_obs_states = num_classes

    def get_obs_encoding(logits):
        """

        Args:
            logits:

        Returns: np array

        """
        if obs=="buckets":

            preds = torch.nn.Softmax(dim=1)(logits).numpy(force=True)
            reduced_preds = logits.argmax(1).type(torch.int).numpy(force=True)

            split_points = np.array(list(range(0, n_buckets + 1))) / n_buckets
            split_points[0] = -1
            split_points[-1] = 2
            # convert predicitons to observations
            curr_pred1d = []
            for j in range(preds.shape[0]):
                preds[j] = np.digitize(preds[j], split_points) - 1
                pred = tuple([reduced_preds[j], tuple(preds[j])])
                curr_pred1d.append(combinations.index(pred))
            preds1d = np.array(curr_pred1d)
        elif obs=="top_k":
            preds = torch.nn.Softmax(dim=1)(logits).numpy(force=True)
            preds1d = preds.argsort(axis=1)[:, -top_k:]
            curr_pred1d = []
            for j in range(preds.shape[0]):
                pred = tuple(preds1d[j])
                curr_pred1d.append(combinations.index(pred))
            preds1d = np.array(curr_pred1d)
        elif obs=="top_k_probs":
            split_points = np.array(list(range(0, n_buckets + 1))) / n_buckets
            split_points[0] = -1
            split_points[-1] = 2
            preds = torch.nn.Softmax(dim=1)(logits).numpy(force=True)
            top_preds_indices = preds.argsort(axis=1)[:, -top_k:]
            top_preds_probs = np.sort(preds, axis=1)[:, -top_k_probs:]
            curr_pred1d = []
            for j in range(preds.shape[0]):
                pred_topk = tuple(top_preds_indices[j])
                pred_probs = top_preds_probs[j]
                pred_probs = tuple(np.digitize(pred_probs, split_points) - 1)
                pred = tuple([pred_probs, pred_topk])
                curr_pred1d.append(combinations.index(pred))
            preds1d = np.array(curr_pred1d)
        else:
            preds1d = logits.argmax(1).type(torch.int).numpy(force=True)

        return preds1d

    def invert_encoding(preds):
        if obs=="buckets":
            get_argmax = np.vectorize(lambda x: combinations[x][0])
            preds = get_argmax(preds)
        elif obs=="top_k":
            get_argmax = np.vectorize(lambda x: combinations[x][-1])
            preds = get_argmax(preds)
        elif obs=="top_k_probs":
            get_argmax = np.vectorize(lambda x: combinations[x][-1][-1])
            preds = get_argmax(preds)
        else:
            preds = preds
        return preds

    preds_obsstate = [get_obs_encoding(x) for x in preds]
    preds_obsstate_train = [get_obs_encoding(x) for x in preds_train]

    if train:
        hmm = None
        print("Not implemented yet")
    else:
        hmm = CategoricalHMM(n_components=num_hidden_states, verbose=True)
        trans, emiss, start = calculate_probabilities(
            labels, preds_obsstate, num_hidden_states, num_obs_states, train_hidden_states=labels_train, train_observations=preds_obsstate_train
        )

        hmm.transmat_ = trans
        hmm.emissionprob_ = emiss
        hmm.startprob_ = start

    return hmm, get_obs_encoding, invert_encoding


def calculate_probabilities(hidden_states, observations, n_states, n_observations, train_hidden_states=None, train_observations=None):
    """
    Args:
        hidden_states: Labels are assumed to be integers starting from 0 and consecutive
        observations: Labels are assumed to be integers starting from 0 and consecutive
        n_states:
        n_observations:

    Returns:

    """
    # Initialize matrices
    transition_prob = np.zeros((n_states, n_states))
    emission_prob = np.zeros((n_states, n_observations))
    starter_prob = np.zeros(n_states)

    # Calculate transition probabilities, if train data is given we additionally use that
    for subject in hidden_states:
        for i in range(len(subject) - 1):
            transition_prob[subject[i], subject[i + 1]] += 1
    if train_hidden_states is not None:
        for subject in train_hidden_states:
            for i in range(len(subject) - 1):
                transition_prob[subject[i], subject[i + 1]] += 1
    transition_prob /= transition_prob.sum(axis=1, keepdims=True)

    # Calculate emission probabilities
    for i in range(len(observations)):
        for j in range(len(observations[i])):
            emission_prob[hidden_states[i][j], observations[i][j]] += 1
    emission_prob /= emission_prob.sum(axis=1, keepdims=True)

    # Calculate starter probabilities
    for subject in hidden_states:
        starter_prob[subject[0]] += 1
    if train_hidden_states is not None:
        for subject in train_hidden_states:
            starter_prob[subject[0]] += 1
    starter_prob /= starter_prob.sum()

    return transition_prob, emission_prob, starter_prob
