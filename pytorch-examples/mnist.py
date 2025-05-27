import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch
from torch import nn
from pathlib import Path


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
    torch.manual_seed(0)
    # data
    data_dir = Path(__file__).resolve().parent / "data"
    data_dir.mkdir(exist_ok=True)

    train_mnist = MNIST(
        root=data_dir,
        download=True,
        train=True,
        transform=transforms.ToTensor(),
    )
    test_mnist = MNIST(
        root=data_dir,
        download=True,
        train=False,
        transform=transforms.ToTensor(),
    )

    train_mnist = Subset(train_mnist, range(5000))
    test_mnist = Subset(test_mnist, range(5000))

    epochs = 2
    batch_size = 32

    train_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False)

    # model - as small as possible since micrograd_rs is not optimized
    model = nn.Sequential(
        nn.Flatten(1),
        nn.Linear(28 * 28, 10, bias=True),
    )
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    # train
    for e in range(epochs):
        for batch_data, batch_labels in train_loader:
            logits = model.forward(batch_data)
            loss = F.cross_entropy(logits, batch_labels)

            optim.zero_grad()
            loss.backward()
            optim.step()

        train_acc = score(model, train_loader)
        test_acc = score(model, test_loader)
        print(f"Epoch {e}, Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")


if __name__ == "__main__":
    main()
