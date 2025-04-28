import torch
from tqdm import tqdm
import math

def classification_train_loop(dataloader, model, device, loss_fn, optimizer):
    model.train()
    loss_total = None
    for batch, (X, y) in enumerate(progress := tqdm(dataloader)):
        optimizer.zero_grad()
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)


        # Backpropagation

        loss.backward()
        optimizer.step()

        loss_total = loss_total + loss if loss_total is not None else loss
    loss_total = loss_total.item()
    return loss_total


def classification_test_loop(dataloader, model, device, loss_fn, num_classes):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    pred_df = []
    target = []
    with torch.no_grad():
        for X, y in tqdm(dataloader, mininterval=10):
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum()
            # Per class predictions
            pred_df.extend(pred.argmax(1).type(torch.int).tolist())
            target.extend(y.numpy(force=True).tolist())
        test_loss = test_loss.item()
        test_loss /= num_batches

        '''correct = correct.item()
        correct /= size
        balanced_accuracy = multiclass_accuracy(
            torch.tensor(pred_df), torch.tensor(target), num_classes, average="macro"
        )
        pred_df = pd.DataFrame(data=pred_df, columns=["Predicted Class"])
        pred_df["count"] = 1
        pred_df = pred_df.groupby("Predicted Class").count()
        # print(pred_df)'''

    return test_loss

def split_subject_wise(data, test_ratio, random_gen):
    """
    Splits data into train and test set subject-wise.
    :param data: List of data containers, one container per subject
    :param test_ratio: Percentage of data that should be in the test set. Number of elements in test set is rounded up to the next integer.
    :param random_gen: Seeded random number generator
    :return:
    """
    test_length = math.ceil(len(data) * test_ratio)
    indices = list(range(len(data)))
    random_gen.shuffle(indices)
    test_indices = indices[0:test_length]
    train_indices = indices[test_length:]
    test = [data[i] for i in test_indices]
    train = [data[i] for i in train_indices]
    return train, test