import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

# 32 x 32 grid
image_size = 32

# This contains the concentration value for each pixel on a 32 x 32 grid that start random but will be optimised later
C = torch.randn(image_size, image_size, requires_grad=True)

# This will create coordinates for each pixel so each pixel knows where it is
xs = torch.linspace(0, 1, image_size)
ys = torch.linspace(0, 1, image_size)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

cx, cy = 0.5, 0.5 # Centre of blob
sigma = 0.05 # Gives the width of the blob

# Target image: (Centre blob in the middle to optimise to)
b = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * sigma**2))

lr = 0.5 # Learning rate of the optimiser
n_iters = 200 # Number of iterations

for i in range(n_iters):
    """This will repeat the optimisation process until the target (real) image matches the predicted image"""

    # Temporary Forward model: (Finds the predicted image)
    pred = torch.sigmoid(C)  # This ensures that the values will always be betwen 0 and 1

    loss = torch.norm(pred - b)**2 # Uses the equation to measure the difference between the predicted and target image

    loss.backward() # This computes the gradients

    #Performs the gradient descent step:
    with torch.no_grad():
        C -= lr * C.grad

    C.grad.zero_() # Clears the gradients for the next iteration so gradients dont add up for no reason

    if i % 20 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# show result
plt.imshow(pred.detach().numpy())
plt.title("Learned Concentration")
plt.colorbar()
plt.show()