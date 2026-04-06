import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# image size
image_size = 32

# concentration field (this is what we're learning)
C = torch.randn(image_size, image_size, requires_grad=True)

xs = torch.linspace(0, 1, image_size)
ys = torch.linspace(0, 1, image_size)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

cx, cy = 0.5, 0.5
sigma = 0.05

b = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * sigma**2))

# optimiser settings
lr = 0.5
n_iters = 200

for i in range(n_iters):

    # forward (now trivial)
    pred = torch.sigmoid(C)  # ensure non-negative concentration

    # loss
    loss = torch.norm(pred - b)**2

    # backward
    loss.backward()

    # update
    with torch.no_grad():
        C -= lr * C.grad

    C.grad.zero_()

    if i % 20 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# show result
plt.imshow(pred.detach().numpy())
plt.title("Learned Concentration")
plt.colorbar()
plt.show()