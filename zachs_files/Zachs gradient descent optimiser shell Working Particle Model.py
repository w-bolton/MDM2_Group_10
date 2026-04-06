import torch
torch.manual_seed(0)

n_particles = 20 # Im modelling 20 particles

# Theta contains all the values for: [x positions, y positions, vx velocities, vy velocities, density]
theta = torch.randn(n_particles, 5, requires_grad=True)

# This creates a target image (placeholder):
image_size = 32
b = torch.zeros(image_size, image_size)
b[16, 16] = 5.0 # This produces a bright spot in the middle of the image

# Temporary Forward Model (Had to make in order for code to work):
# Fully generated this forward model using AI as Qusai is making a proper one:
def A_model(theta):
    """Forward model"""

    # keep values stable
    x = torch.sigmoid(theta[:, 0])
    y = torch.sigmoid(theta[:, 1])
    vx = theta[:, 2]
    vy = theta[:, 3]
    density = torch.sigmoid(theta[:, 4]) #Densities over 0

    # evolve
    dt = 0.1
    x_new = x + vx * dt
    y_new = y + vy * dt

    # grid
    xs = torch.linspace(0, 1, image_size)
    ys = torch.linspace(0, 1, image_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

    image = torch.zeros(image_size, image_size, device=theta.device)

    sigma = 0.1

    for i in range(len(x_new)):
        dx = grid_x - x_new[i]
        dy = grid_y - y_new[i]
        dist2 = dx**2 + dy**2

        blob = torch.exp(-dist2 / (2 * sigma**2))

        image += density[i] * blob

    return image # Predicted Image


# Gradient Descent Optimiser:
lr = 0.05 # Learning rate of the optimiser
n_iters = 300 # Number of iterations

for i in range(n_iters):
    """This will repeat the optimisation process until the target (real) image matches the predicted image"""

    pred = A_model(theta) # Uses the forward model to predict an image
    loss = torch.norm(pred - b)**2 # Uses the equation to measure the difference between the predicted and target image

    loss.backward() # This computes the gradients

    #Performs the gradient descent step
    with torch.no_grad():
        theta -= lr * theta.grad

    theta.grad.zero_() # Clears teh gradients for the next iteration so gradients dont add up for no reason

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

print("\nFinal Results:")

# Extracts the final optimised values:
x = torch.sigmoid(theta[:, 0])
y = torch.sigmoid(theta[:, 1])
vx = theta[:, 2]
vy = theta[:, 3]
density = torch.relu(theta[:, 4])

for i in range(n_particles):
    print(f"\nParticle {i}:")
    print(f"  x position = {x[i].item():.4f}")
    print(f"  y position = {y[i].item():.4f}")
    print(f"  velocity x = {vx[i].item():.4f}")
    print(f"  velocity y = {vy[i].item():.4f}")
    print(f"  density = {density[i].item():.4f}")

import matplotlib.pyplot as plt

plt.imshow(pred.detach().numpy())
plt.title("Predicted Image")
plt.colorbar()
plt.show()