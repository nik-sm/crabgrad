import torch
import torch.nn.functional as F
from torch.optim import SGD


def main():
    # See the effect of a single step using SGD when directly maximizing a single class' probability
    logits = torch.tensor([[1.0, 1.0]], requires_grad=True)  # "parameters"
    y_true = torch.tensor([0])

    probs_before = torch.log_softmax(logits, dim=-1).exp()
    print(f"before: {logits.tolist()=}, {probs_before.tolist()=}")

    loss = F.nll_loss(logits, y_true)
    optim = SGD([logits])
    optim.zero_grad()
    loss.backward()
    optim.step()

    probs_after = torch.log_softmax(logits, dim=-1).exp()
    print(f"after: {logits.tolist()=}, {probs_after.tolist()=}")


if __name__ == "__main__":
    main()
