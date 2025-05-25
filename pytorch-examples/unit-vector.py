import torch
from torch.optim import SGD


def main():
    # Example of optimizing a single layer and seeing that it will move towards being aligned with a given unit vector
    layer = torch.nn.Linear(3, 1, bias=False)
    data = torch.tensor([1.0, 0.0, 0.0])
    optim = SGD(layer.parameters(), lr=0.1)

    print(f"before: {layer.weight}")

    for _ in range(1000):
        out = layer.forward(data)
        loss = 1 - out

        optim.zero_grad()
        loss.backward()
        optim.step()

        # Normalize to unit length
        with torch.no_grad():
            layer.weight /= layer.weight.pow(2).sum().sqrt()

    print(f"after: {layer.weight}")

    assert torch.allclose(layer.weight, data)


if __name__ == "__main__":
    main()
