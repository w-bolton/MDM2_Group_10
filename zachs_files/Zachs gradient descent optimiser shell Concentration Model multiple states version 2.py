import torch
import matplotlib.pyplot as plt
import time

torch.manual_seed(0)

# 32 x 32 grid
image_size = 32

n_steps = 5
lr = 0.5 # Learning rate of the optimiser
n_iters = 200 # Number of iterations
lambda_dyn = 1.0  # Strength of physics constraint

# This contains the concentration value for each pixel on a 32 x 32 grid that start random but will be optimised later
C = torch.randn(image_size, image_size, requires_grad=True)

# This will create coordinates for each pixel so each pixel knows where it is
xs = torch.linspace(0, 1, image_size)
ys = torch.linspace(0, 1, image_size)
grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

cx, cy = 0.5, 0.5 # Centre of blob
sigma = 0.05 # Gives the width of the blob

#sigma_data = 0.1
#sigma_dyn = 0.2
#loss = (1/sigma_data**2)*data_loss + (1/sigma_dyn**2)*dyn_loss

# Target image: (Centre blob in the middle to optimise to):
b = torch.exp(-((grid_x - cx)**2 + (grid_y - cy)**2) / (2 * sigma**2))

# Simple Advection function to be subbed into:
def advect(C_t):
    """
    Moves concentration slightly to the right (toy physics)
    """
                                                                            #This is where the advect code goes using this line here: C_next = linns_advection_model(C_t)
    return torch.roll(C_t, shifts=1, dims=1)


# Optimisation:
for i in range(n_iters):
    """This will repeat the optimisation process until the target (real) images matches the predicted images"""

    # Data term
    data_loss = 0
    for t in range(n_steps):
        pred = torch.sigmoid(C[t])                                           #This is the forward model so replace this with actual model: pred = forward_model(C)
        data_loss += torch.norm(pred - b)**2

    # Dynamics Term:
    dyn_loss = 0
    for t in range(n_steps - 1):
        C_next_pred = advect(C[t])   # predicted next state
        dyn_loss += torch.norm(C[t+1] - C_next_pred)**2

    # Total Loss:
    loss = data_loss + lambda_dyn * dyn_loss

    # Back propogation:
    loss.backward()

    with torch.no_grad():
        C -= lr * C.grad

    C.grad.zero_()

    if i % 20 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

# Plotting results:
print("\nPlaying animation:")

plt.figure()

for t in range(n_steps):
    plt.clf()  # This clears the previous frame
    plt.imshow(torch.sigmoid(C[t]).detach().numpy())
    plt.title(f"Time {t}")
    plt.colorbar()
    plt.pause(1)

plt.show()