import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def make_binary_classification(n_samples_each_class, n_features):
    torch.manual_seed(0)
    data, labels = [], []

    # class 0
    data.append(
        torch.randn(n_samples_each_class * n_features).reshape(
            n_samples_each_class, n_features
        )
    )
    labels.extend([0] * n_samples_each_class)

    # class 1
    data.append(
        torch.randn(n_samples_each_class * n_features)
        .add(2)
        .reshape(n_samples_each_class, n_features)
    )
    labels.extend([1] * n_samples_each_class)

    data = torch.concat(data)
    labels = torch.tensor(labels)
    return data, labels


def train_test_split(data, labels, train_frac, test_frac):
    torch.manual_seed(0)
    n_total = len(labels)
    n_train = int(train_frac / (train_frac + test_frac) * n_total)

    _rand_idx = torch.randperm(n_total)

    train_data, train_labels = data[_rand_idx[:n_train]], labels[_rand_idx[:n_train]]
    test_data, test_labels = data[_rand_idx[n_train:]], labels[_rand_idx[n_train:]]

    return (train_data, train_labels), (test_data, test_labels)


@torch.no_grad
def score(model, loader):
    n_total = 0
    n_correct = 0
    for x, y in loader:
        logits = model.forward(x)
        correct = logits.argmax(-1).eq(y).sum()
        n_total += len(y)
        n_correct += correct

    return n_correct / n_total


def main():
    n_features = 64
    n_classes = 2
    n_samples_each_class = 1000
    epochs = 10
    batch_size = 32

    data, labels = make_binary_classification(
        n_samples_each_class=n_samples_each_class, n_features=n_features
    )
    (train_data, train_labels), (test_data, test_labels) = train_test_split(
        data, labels, 0.8, 0.2
    )

    model = nn.Sequential(
        nn.Linear(n_features, 32, bias=True),
        nn.ReLU(inplace=True),
        nn.Linear(32, n_classes, bias=True),
    )
    train_loader = DataLoader(
        TensorDataset(train_data, train_labels), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_data, test_labels), batch_size=batch_size, shuffle=True
    )
    optim = AdamW(
        model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    for epoch in range(epochs):
        for batch_data, batch_labels in train_loader:
            logits = model.forward(batch_data)
            loss = F.cross_entropy(logits, batch_labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_acc = score(model, train_loader)
        test_acc = score(model, test_loader)

        print(f"Train acc: {train_acc}")
        print(f"Test acc: {test_acc}")


if __name__ == "__main__":
    main()
