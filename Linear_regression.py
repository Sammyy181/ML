import numpy as np
import matplotlib.pyplot as plt
import random


#Define start and end points of line
x1 = 0
y1 = random.randint(0,10)

x2 = 1
y2 = random.randint(0,10)

print(f"{x1} {x2} {y1} {y2}")

def gen_points(n): 
    #Generate n random points
    dists = np.random.rand(n)
    #print(dists)
    
    points = []
    
    for pt in range (0,n):
        pt_x = x2 * (1 - dists[pt])
        pt_y = (y1 * dists[pt]) + (y2 * (1 - dists[pt]))
        points.append((pt_x,pt_y))
    
    noisy_points = [(x, y + np.random.normal(0, 0.5)) for x, y in points]
    #print(f"Points - {points}")
    
    return noisy_points

def plot_graph(points, theta):
    #Plots graph with the points, and the original line
    plt.plot((x1,x2), (y1,y2), linestyle='-', color='red')
    
    #Plotting the estimated line
    x_estim = np.array([0,1])
    y_estim = theta[0] + (theta[1] * x_estim)
    plt.plot(x_estim,y_estim, linestyle='-', color='green',label = "Estimated line")
    
    x, y = zip(*points)
    plt.scatter(x, y, color='blue', marker='o')  # Scatter plot
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Points")
    plt.grid(True)  # Adds grid for better visualization
    plt.show()
    
def plot_loss(losses):
    epochs_0 = range(1, 1 + 10 * len(losses), 10)  # X-axis: Epoch numbers
    epochs = list(epochs_0)
    plt.plot(epochs, losses, marker='o', linestyle='-', color='blue', label="Loss Curve")

    # Labels and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training Loss Over Time")
    
    plt.grid(True)
    plt.legend()
    plt.show()

def get_loss(theta, points):
    
    loss = 0
    n = len(points)
    
    for point in points:
        loss += ((theta[0] + (theta[1]*point[0])) - point[1]) ** 2
        
    return loss/n

def descent(theta, alpha, points):
    
    new_theta = np.copy(theta)  # Create a copy to prevent overwriting during updates
    n = len(points)

    sum_error_0 = 0
    sum_error_1 = 0

    for point in points:
        val = new_theta[0] + (new_theta[1] * point[0])
        error = val - point[1]
        sum_error_0 += error
        sum_error_1 += error * point[0]

    # Update using mean gradients
    new_theta[0] -= (alpha / n) * sum_error_0
    new_theta[1] -= (alpha / n) * sum_error_1

    return new_theta
        

def main(num_points,num_iter, lim_loss):
    
    # Initialise m and c 
    # The function is y = mx + c

    m = random.uniform(-10,10)
    c = random.uniform(0,10)

    #Considering the function y = mx + c as h(x) = theta0 + theta1 * x
    theta = np.array([c,m])
    
    #0.075 for normal, 0.2 for fast, 1e-3 for very slow, 2 for divergence
    alpha = 2
    
    sample_points = gen_points(num_points)
    
    square_loss = get_loss(theta,sample_points)
    
    loss_history = []
    
    epoch_count = 0
    
    while (epoch_count < num_iter and square_loss > lim_loss) :
        
        theta = descent(theta, alpha, sample_points)
        epoch_count += 1
        square_loss = get_loss(theta, sample_points)
        if(epoch_count % 10 == 1):
            loss_history.append(square_loss)
            print(f"Epoch {epoch_count}: Loss = {square_loss}")

    if (epoch_count == num_iter): print(f"Completed {num_iter} Iterations")
    else: print(f"Achieved loss below {lim_loss}")
    
    plot_graph(sample_points,theta)
    plot_loss(loss_history)
    
    
if __name__ == "__main__":
    main(20,100,0.1)