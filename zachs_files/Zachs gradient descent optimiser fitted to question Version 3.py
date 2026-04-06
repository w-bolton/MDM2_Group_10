import torch
torch.manual_seed(0)

# --- PARAMETERS (now more meaningful) ---
n_particles = 20

# theta now contains:
# [x positions, y positions, vx velocities, vy velocities, density]
theta = torch.randn(n_particles, 5, requires_grad=True)

# target "image" (placeholder)
image_size = 32
b = torch.zeros(image_size, image_size)
b[16, 16] = 5.0

# --- MODEL ---
def A_model(theta):

    # keep values stable
    x = torch.sigmoid(theta[:, 0])
    y = torch.sigmoid(theta[:, 1])
    vx = theta[:, 2]
    vy = theta[:, 3]
    density = torch.relu(theta[:, 4])

    # evolve
    dt = 0.1
    x_new = x + vx * dt
    y_new = y + vy * dt

    # grid
    xs = torch.linspace(0, 1, image_size)
    ys = torch.linspace(0, 1, image_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')

    image = torch.zeros(image_size, image_size, device=theta.device)

    sigma = 0.05

    for i in range(len(x_new)):
        dx = grid_x - x_new[i]
        dy = grid_y - y_new[i]
        dist2 = dx**2 + dy**2

        blob = torch.exp(-dist2 / (2 * sigma**2))

        image += density[i] * blob

    return image


# --- OPTIMISER ---
lr = 1e-3
n_iters = 100

for i in range(n_iters):

    pred = A_model(theta)
    loss = torch.norm(pred - b)**2

    loss.backward()

    with torch.no_grad():
        theta -= lr * theta.grad

    theta.grad.zero_()

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

print("\nFinal Results:")

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
