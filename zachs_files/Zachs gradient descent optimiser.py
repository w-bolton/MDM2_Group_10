import numpy as np

np.random.seed(0) #Keeps the coding model the same
W = np.random.randn(10, 5) #Place holder model:

def A_model(theta):
    """Replace this function later with actual model: This just uses A(θ) = Wθ to get a placeholder image"""
    return W @ theta

def loss_fn(pred, b):
    """Measures how wrong the prediction is using the given equation: L = (∥A(θ)-b∥)^2"""
    return np.linalg.norm(pred - b)**2

def numerical_grad(theta, model, b, eps = 1e-5):
    """This will approximate a value to ∂L/∂θi"""
    grad = np.zeros_like(theta)
    for i in range(len(theta)): #Loops over each paramter ∂θi
        #This will create 2 different versions so that you can easily find the gradient between 2 points:
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        
        #This will find a value of theta just above and below the point on the line in order to find the difference later on:
        theta_plus[i] = theta_plus[i] + eps
        theta_minus[i] = theta_minus[i] - eps

        #Runs the forward model on each value of theta to find the losses
        loss_plus = loss_fn(model(theta_plus), b)
        loss_minus = loss_fn(model(theta_minus), b)

        grad[i] = (loss_plus - loss_minus) / (2 * eps) #Central difference approximation: estimates slope of the loss curve

    return grad

def gradient_descent(theta, model, b, lr = 1e-4, n_iters = 100):
    for i in range(n_iters): #This will gradually improve theta over time
        pred = model(theta)
        loss = loss_fn(pred, b)
        grad = numerical_grad(theta, model, b)

        if not np.isfinite(loss): #If something goes wrong this will stop it
            print("Loss exploded.")
            break

        if not np.all(np.isfinite(grad)): #If the gradient has nvalid values this will stop the loop
            print("Gradient exploded.")
            break

        theta = theta - (lr * grad) #This will move theta in the idrection that reduces the loss

        if i % 10 == 0:
            print(f"Iteration {i}, Loss: {loss:.6f}") #This will print progress every 10 iterations

    return theta


theta = np.random.randn(5)
b = np.random.randn(10)

theta_opt = gradient_descent(theta, A_model, b)

print("Optimized theta:", theta_opt)
