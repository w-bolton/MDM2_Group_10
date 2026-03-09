import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
image_size = 200
num_particles = 150
absorption_coeff = 0.02
particle_intensity = 1.0
dye_intensity = 0.5

# Create grid (camera sensor)
x = np.linspace(-1, 1, image_size)
y = np.linspace(-1, 1, image_size)
X, Y = np.meshgrid(x, y)

# distance from light source
distance = np.sqrt(X**2 + Y**2)

# Beer-Lambert attenuation
attenuation = np.exp(-absorption_coeff * distance)

# Example dye concentration field
dye_concentration = np.exp(-(X**2 + Y**2) * 3)

# fluorescence intensity
dye_image = dye_intensity * dye_concentration * attenuation

# Generate random particles
particles = np.random.uniform(-1, 1, (num_particles, 2))

particle_image = np.zeros((image_size, image_size))

# render particles as small Gaussian spots
for px, py in particles:
    dx = X - px
    dy = Y - py
    particle_image += particle_intensity * np.exp(-(dx**2 + dy**2) / 0.001)

# apply attenuation
particle_image *= attenuation

# Final image
image = particle_image + dye_image

# normalize
image = image / image.max()

# Display result
plt.figure(figsize=(6,6))
plt.imshow(image, cmap="inferno", extent=[-1,1,-1,1])
plt.title("Simulated Optical Image")

# Add grid
plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xlabel("X Position")
plt.ylabel("Y Position")
# Define grid spacing
grid_spacing = 0.1   # smaller number = smaller boxes

ticks = np.arange(-1, 1 + grid_spacing, grid_spacing)

plt.xticks(ticks)
plt.yticks(ticks)

plt.grid(True, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
plt.show()