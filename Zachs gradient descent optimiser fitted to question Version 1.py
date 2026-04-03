import torch
torch.manual_seed(0)

# --- PARAMETERS (now more meaningful) ---
n_particles = 20

# theta now contains:
# [x positions, y positions, vx velocities, vy velocities, density]
theta = torch.randn(n_particles, 5, requires_grad=True)

# target "image" (placeholder)
image_size = 32
b = torch.randn(image_size, image_size)

# --- MODEL ---
def A_model(theta):
    """
    Maps particle state → image
    """

    # unpack theta
    x = theta[:, 0]
    y = theta[:, 1]
    vx = theta[:, 2]
    vy = theta[:, 3]
    density = theta[:, 4]

    # --- evolve particles (simple physics placeholder) ---
    dt = 0.1
    x_new = x + vx * dt
    y_new = y + vy * dt

    # --- render to image ---
    image = torch.zeros(image_size, image_size)

    for i in range(len(x_new)):
        xi = torch.clamp((x_new[i] * image_size).long(), 0, image_size-1)
        yi = torch.clamp((y_new[i] * image_size).long(), 0, image_size-1)

        image[xi, yi] += density[i]

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

x = theta[:, 0]
y = theta[:, 1]
vx = theta[:, 2]
vy = theta[:, 3]
density = theta[:, 4]

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