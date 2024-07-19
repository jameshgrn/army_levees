#%%
import numpy as np
import matplotlib.pyplot as plt

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def softmax(z):
    e_z = np.exp(z - np.max(z))  # for numerical stability
    return e_z / e_z.sum(axis=0)

def calculate_slope(e_c, e_n, dx, dy, no_data_value):
    if e_c == no_data_value or e_n == no_data_value or (e_c == e_n == no_data_value):
        return np.NINF
    if dx == dy:  # Diagonal move
        return (e_c - e_n) / np.sqrt(2)
    else:  # Orthogonal move
        return e_c - e_n

def calculate_inertia(V_last, V_direction):
    return (1 + cosine_similarity(V_last, V_direction)) / 2

def calculate_erosion(S_ij, alpha):
    return np.tanh(alpha * S_ij) if S_ij > 0 else 0

def calculate_deposition(S_ij, beta):
    return np.tanh(-beta * S_ij) if S_ij < 0 else 0

def compute_transition_probabilities(x, y, neighbors, elevations, V_last, phi, psi, epsilon, delta, no_data_value):
    z_vectors = []
    for (i, j) in neighbors:
        dx, dy = abs(i - x), abs(j - y)
        e_c = elevations[x, y]
        e_n = elevations[i, j]
        S_ij = calculate_slope(e_c, e_n, dx, dy, no_data_value)
        I_ij = calculate_inertia(V_last, [i - x, j - y])
        # Directly use S_ij for erosion and deposition calculations
        E_ij = np.tanh(S_ij) if S_ij > 0 else 0  # Erosion effect
        D_ij = np.tanh(-S_ij) if S_ij < 0 else 0  # Deposition effect
        z_ij = np.array([phi * S_ij, psi * I_ij, epsilon * E_ij, -delta * D_ij])
        z_vectors.append(np.sum(z_ij))  # Combine the influences into a single value for softmax
    
    probabilities = softmax(np.array(z_vectors))
    return dict(zip(neighbors, probabilities))

def create_gaussian_bump(size, height):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x+y*y)
    sigma, mu = 0.4, 0.0
    return height * np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

def initialize_walker_position(elevations):
    mid_x, mid_y = np.array(elevations.shape) // 2
    return mid_x, mid_y

def get_neighbors(x, y, size, last_position=None):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size:
                if last_position is None or (nx, ny) != last_position:
                    neighbors.append((nx, ny))
    return neighbors

def create_noisy_gaussian_bump(size, height, noise_level):
    bump = create_gaussian_bump(size, height)
    noise = np.random.normal(0, noise_level, bump.shape)
    noisy_bump = bump + noise
    return noisy_bump

def simulate_walker(steps, size, height, phi, psi, epsilon, delta, no_data_value, noise_level, initial_elevations=None):
    if initial_elevations is None:
        initial_elevations = create_noisy_gaussian_bump(size, height, noise_level)
    elevations = np.copy(initial_elevations)
    x, y = initialize_walker_position(elevations)
    V_last = [0, 1]  # Initial direction vector
    path = [(x, y)]
    last_position = None  # Initialize last_position to prevent immediate backtracking

    for step in range(steps):
        neighbors = get_neighbors(x, y, size, last_position)  # Exclude the last position
        probabilities = compute_transition_probabilities(x, y, neighbors, elevations, V_last, phi, psi, epsilon, delta, no_data_value)
        
        next_position = max(probabilities, key=probabilities.get)  # Choose next step based on highest probability
        V_last = [next_position[0] - x, next_position[1] - y]  # Update direction vector
        
        S_ij = calculate_slope(elevations[x, y], elevations[next_position[0], next_position[1]], abs(next_position[0] - x), abs(next_position[1] - y), no_data_value)
        # Adjust elevation based on slope
        if S_ij > 0:
            elevations[x, y] -= epsilon * np.tanh(S_ij)  # Erosion effect
        elif S_ij < 0:
            elevations[x, y] += delta * np.tanh(-S_ij)  # Deposition effect
        
        elevations[x, y] = max(elevations[x, y], 0)  # Ensure elevation doesn't go below 0
        
        last_position = (x, y)  # Update last_position before moving
        x, y = next_position
        path.append((x, y))
    
    return path, elevations

def simulate_multiple_walkers(num_walkers, steps, size, height, phi, psi, epsilon, delta, no_data_value, noise_level, initial_elevations=None):
    if initial_elevations is None:
        initial_elevations = create_noisy_gaussian_bump(size, height, noise_level)
    elevations = np.copy(initial_elevations)
    paths = []
    
    for _ in range(num_walkers):
        path, elevations = simulate_walker(steps, size, height, phi, psi, epsilon, delta, no_data_value, noise_level, elevations)
        paths.append(path)
    
    return paths, elevations, initial_elevations

# Visualization
def visualize_paths(paths, final_elevations, initial_elevations):
    plt.figure(figsize=(10, 8))
    plt.imshow(final_elevations, cmap='terrain', alpha=0.6)
    plt.colorbar(label='Elevation')
    
    for path in paths:
        path_x, path_y = zip(*path)
        plt.plot(path_y, path_x, linewidth=2)
    
    plt.title('Multiple Walker Paths on Noisy Gaussian Bump')
    plt.show()

    # Visualization of the elevation changes due to erosion and deposition
    elevation_changes = final_elevations - initial_elevations
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_changes, cmap='coolwarm', alpha=0.6)
    plt.colorbar(label='Elevation Change')
    plt.title('Elevation Changes due to Erosion and Deposition on Noisy Terrain')
    plt.show()


if __name__ == '__main__':
    # Parameters
    num_walkers = 2000
    steps = 100
    size = 150
    height = 150
    # Update to only include the four parameters of the simplified model
    phi, psi, epsilon, delta = 1, .2, .1, 0.1
    no_data_value = -9999
    noise_level = 5

    # Run the simulation for multiple walkers
    # Ensure the function call matches the updated parameter list.
    paths, final_elevations, initial_elevations = simulate_multiple_walkers(
        num_walkers=num_walkers, 
        steps=steps, 
        size=size, 
        height=height, 
        phi=phi, 
        psi=psi, 
        epsilon=epsilon, 
        delta=delta, 
        no_data_value=no_data_value, 
        noise_level=noise_level
    )

    # Visualize the paths and elevation changes
    visualize_paths(paths, final_elevations, initial_elevations)
    
    
#%%
import numpy as np

def simulate_walker(size, height, phi, psi, epsilon, delta, no_data_value, noise_level, initial_lambda, lambda_threshold=2, initial_elevations=None):
    if initial_elevations is None:
        initial_elevations = create_noisy_gaussian_bump(size, height, noise_level)
    elevations = np.copy(initial_elevations)
    x, y = initialize_walker_position(elevations)
    V_last = [0, 1]  # Initial direction vector
    path = [(x, y)]
    last_position = None  # Initialize last_position to prevent immediate backtracking
    lambda_value = initial_lambda

    while lambda_value < lambda_threshold:
        neighbors = get_neighbors(x, y, size, last_position)  # Exclude the last position
        # Remove lambda_value from the arguments
        probabilities = compute_transition_probabilities(x, y, neighbors, elevations, V_last, phi, psi, epsilon, delta, no_data_value)
        
        next_position = max(probabilities, key=probabilities.get)  # Choose next step based on highest probability
        V_last = [next_position[0] - x, next_position[1] - y]  # Update direction vector
        
        # Update position based on direction vector
        x, y = x + V_last[0], y + V_last[1]
        path.append((int(x), int(y)))  # Append new position to path
        
        # Ensure the walker stays within bounds
        x, y = max(0, min(size - 1, int(x))), max(0, min(size - 1, int(y)))

        # Update lambda based on some growth function or external influence
        lambda_value += 0.01  # Example increment, adjust based on your model dynamics

    return path, elevations

# Parameters
initial_lambda = 0.1  # Starting value of lambda

# Simulation call
path, elevations = simulate_walker(
    size=150, height=150, phi=1, psi=2, epsilon=0.1, delta=0.1,
    no_data_value=-9999, noise_level=5, initial_lambda=initial_lambda
)
# %%
