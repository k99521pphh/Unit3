import numpy as np
import matplotlib.pyplot as plt
import os

# Constants for the map
MAP_WIDTH = 50
MAP_HEIGHT = 45
POPULATION_SIZE = 20
GENERATIONS = 50
MUTATION_RATE = 0.1
OUTPUT_FOLDER = "output"

# Terrain types
TERRAIN_TYPES = {
    0: "Empty",
    1: "Grass",
    2: "Water",
    3: "Rock",
    4: "Forest"
}

# Initialize a random map
def generate_random_map():
    return np.random.randint(0, len(TERRAIN_TYPES), (MAP_HEIGHT, MAP_WIDTH))

# Fitness function
def calculate_fitness(map_grid):
    river_score = evaluate_river(map_grid)
    valley_score = evaluate_valley(map_grid)
    forest_score = evaluate_forest_distribution(map_grid)
    playable_score = evaluate_playable_zones(map_grid)

    # Weights for the fitness components
    w1, w2, w3, w4 = 1.0, 1.0, 0.5, 0.5

    return w1 * river_score + w2 * valley_score + w3 * forest_score + w4 * playable_score

# Evaluate continuous river path
def evaluate_river(map_grid):
    water_positions = np.argwhere(map_grid == 2)
    if len(water_positions) == 0:
        return 0

    # Check if water tiles form a connected path (simple row/column adjacency check)
    connected = True
    for i in range(1, len(water_positions)):
        if not np.any(np.abs(water_positions[i] - water_positions[i - 1]) == 1):
            connected = False
            break

    return 1 if connected else 0

# Evaluate valley shape (grass and rocks forming valleys)
def evaluate_valley(map_grid):
    valley_score = np.sum((map_grid == 1) | (map_grid == 3)) / (MAP_WIDTH * MAP_HEIGHT)
    return valley_score

# Evaluate forest distribution
def evaluate_forest_distribution(map_grid):
    forest_tiles = np.sum(map_grid == 4)
    return forest_tiles / (MAP_WIDTH * MAP_HEIGHT)

# Evaluate playable zones (walkable grass areas)
def evaluate_playable_zones(map_grid):
    grass_tiles = np.sum(map_grid == 1)
    return grass_tiles / (MAP_WIDTH * MAP_HEIGHT)

# Mutate a map
def mutate(map_grid):
    new_map = map_grid.copy()
    for _ in range(int(MAP_WIDTH * MAP_HEIGHT * MUTATION_RATE)):
        x, y = np.random.randint(0, MAP_HEIGHT), np.random.randint(0, MAP_WIDTH)
        new_map[x, y] = np.random.randint(0, len(TERRAIN_TYPES))
    return new_map

# Crossover between two maps
def crossover(map1, map2):
    split = np.random.randint(0, MAP_WIDTH)
    child = np.hstack((map1[:, :split], map2[:, split:]))
    return child

# Generate output folder
def prepare_output_folder():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

# Save map to file
def save_map(map_grid, generation, index):
    filename = os.path.join(OUTPUT_FOLDER, f"map_gen{generation}_idx{index}.txt")
    np.savetxt(filename, map_grid, fmt="%d")

# Visualize and save the map as an image
def save_map_image(map_grid, generation, index):
    plt.imshow(map_grid, cmap="terrain")
    plt.title(f"Map - Gen {generation}, Index {index}")
    plt.axis("off")
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"map_gen{generation}_idx{index}.png"))
    plt.close()

# Main evolutionary algorithm
def evolutionary_algorithm():
    prepare_output_folder()
    population = [generate_random_map() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        fitness_scores = [calculate_fitness(map_grid) for map_grid in population]
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]

        # Save the best maps
        for i, best_map in enumerate(sorted_population[:5]):
            save_map(best_map, generation, i)
            save_map_image(best_map, generation, i)

        # Generate the next generation
        new_population = sorted_population[:POPULATION_SIZE // 2]  # Elitism: keep the top 50%
        while len(new_population) < POPULATION_SIZE:
            indices = np.random.choice(len(sorted_population[:10]), 2, replace=False)
            parent1, parent2 = sorted_population[indices[0]], sorted_population[indices[1]]
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

# Run the algorithm
evolutionary_algorithm()
