import torch
import torch.nn.functional as F
from torch.optim import SGD


def main():

    logits = torch.tensor([[1.0, 1.0]], requires_grad=True)

    lse = torch.logsumexp(logits)
    lse.backward()
    print(f"logsumexp output: {lse}")

    y_true = torch.tensor([0])

    print(f"before: {logits}")
    optim = SGD([logits])

    optim.zero_grad()
    loss = F.cross_entropy(logits, y_true)
    loss.backward()
    optim.step()
    print(f"after: {logits}")


if __name__ == "__main__":
    main()
