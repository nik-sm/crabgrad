import torch
import torch.nn.functional as F
from torch.optim import SGD


def main():
    # Ex1: with identical values, after cross-entropy, do both logits have a non-zero gradient?
    logits = torch.tensor([1.0, 1.0], requires_grad=True)
    y_true = torch.tensor([0])

    print(f"before: {logits}")
    loss = F.cross_entropy(logits, y_true)
    optim = SGD(logits)
    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"after: {logits}")


if __name__ == "__main__":
    main()
