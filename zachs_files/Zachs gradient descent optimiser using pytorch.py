import torch
torch.manual_seed(0) #Makes all results the same

theta = torch.randn(5, requires_grad=True)

W = torch.randn(10, 5) #Placeholder for the model
b = torch.randn(10)

def A_model(theta):
    """Replace this function later with actual model: This just uses A(θ) = Wθ to get a placeholder image"""
    return W @ theta

lr = 1e-3 #Step size
n_iters = 100 #Number of updates made

for i in range(n_iters):

    pred = A_model(theta)
    loss = torch.norm(pred - b)**2

    loss.backward() #This will find dL/dθ

    with torch.no_grad():
        theta -= lr * theta.grad #This will move theta in the idrection that reduces the loss

    theta.grad.zero_() #This resets the gradients so they don’t add up between iterations

    if i % 10 == 0:
        print(f"Iteration {i}, Loss: {loss.item():.6f}")

print("Optimized theta:", theta)
