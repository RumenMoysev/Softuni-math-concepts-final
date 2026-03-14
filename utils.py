import numpy as np
import matplotlib.pyplot as plt

def plot_gradient_descent(step_history: list[float], f: callable, title: str, x_bounds: tuple[float, float] = (-5, 5), path_color: str = "red", path_marker: str = "o") -> None:
    """
    Visualizes the gradient descent path on the cost function curve.
    
    Parameters:
    step_history (list): The history of x values visited.
    f (callable): The cost function.F
    title (str): The title of the graph.
    x_bounds (tuple): The min and max x values for drawing the blue curve.
    path_color (str): The color of the steps.
    path_marker (str): The shape of the step points.
    """
    # 1. Create the data for the smooth curve
    x_curve = np.linspace(x_bounds[0], x_bounds[1], 100)
    y_curve = f(x_curve) 
    y_history = [f(x) for x in step_history]

    # 2. Set up the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_curve, y_curve, label="Cost Function: f(x) = x^2", color="blue", linewidth=2)
    
    # 3. Plot the steps
    plt.plot(step_history, y_history, color=path_color, marker=path_marker, linestyle="dashed", markersize=8, label="Steps")

    # 4. Add labels and formatting
    plt.title(title, fontsize=16)
    plt.xlabel("x (Parameter)", fontsize=12)
    plt.ylabel("f(x) (Cost)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()