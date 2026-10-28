import random

# ---------------------------------------------------------
# 1. DEFINE THE NETWORK PACKET OPTIMIZATION PROBLEM
# ---------------------------------------------------------

# Network paths
# Each path has:
#   delay (ms)
#   packet_loss (%)
#   congestion (%)

paths = [
    {"name": "Path 1", "delay": 20, "loss": 2, "congestion": 30},
    {"name": "Path 2", "delay": 35, "loss": 1, "congestion": 20},
    {"name": "Path 3", "delay": 15, "loss": 4, "congestion": 50},
    {"name": "Path 4", "delay": 25, "loss": 3, "congestion": 25}
]

TOTAL_PACKETS = 100


# ---------------------------------------------------------
# 2. INITIALIZE PARAMETERS
# ---------------------------------------------------------

POPULATION_SIZE = 50
MUTATION_RATE = 0.10
CROSSOVER_RATE = 0.80
GENERATIONS = 100


# ---------------------------------------------------------
# 3. CREATE INITIAL POPULATION
# ---------------------------------------------------------

def create_individual():
    """
    An individual represents the number of packets
    assigned to each network path.

    Example:
    [30, 20, 40, 10]

    means:
    Path 1 -> 30 packets
    Path 2 -> 20 packets
    Path 3 -> 40 packets
    Path 4 -> 10 packets
    """

    individual = [0] * len(paths)

    for _ in range(TOTAL_PACKETS):
        path = random.randint(0, len(paths) - 1)
        individual[path] += 1

    return individual


def create_population():
    return [create_individual() for _ in range(POPULATION_SIZE)]


# ---------------------------------------------------------
# 4. FITNESS FUNCTION
# ---------------------------------------------------------

def fitness(individual):
    """
    Lower network cost = better solution.

    Cost consists of:
        Delay
        Packet Loss
        Congestion
    """

    total_cost = 0

    for i in range(len(paths)):

        packets = individual[i]

        delay = paths[i]["delay"]
        loss = paths[i]["loss"]
        congestion = paths[i]["congestion"]

        # Cost for this path
        path_cost = (
            delay * 0.5
            + loss * 2
            + congestion * 0.3
        )

        total_cost += packets * path_cost

    # Convert cost into fitness.
    # Higher fitness = better solution.
    return 1 / (1 + total_cost)


# ---------------------------------------------------------
# 5. SELECTION
# ---------------------------------------------------------

def selection(population):
    """
    Tournament selection.

    Randomly choose individuals and select
    the one with better fitness.
    """

    tournament_size = 3

    tournament = random.sample(population, tournament_size)

    return max(tournament, key=fitness)


# ---------------------------------------------------------
# 6. CROSSOVER
# ---------------------------------------------------------

def crossover(parent1, parent2):

    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    point = random.randint(1, len(parent1) - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    # Fix packet count
    child1 = normalize(child1)
    child2 = normalize(child2)

    return child1, child2


# ---------------------------------------------------------
# NORMALIZE PACKETS
# ---------------------------------------------------------

def normalize(individual):

    total = sum(individual)

    if total == TOTAL_PACKETS:
        return individual

    # Adjust the largest value
    difference = TOTAL_PACKETS - total

    index = individual.index(max(individual))

    individual[index] += difference

    return individual


# ---------------------------------------------------------
# 7. MUTATION
# ---------------------------------------------------------

def mutation(individual):

    if random.random() < MUTATION_RATE:

        # Select two paths
        path1, path2 = random.sample(range(len(paths)), 2)

        # Move some packets from path1 to path2
        if individual[path1] > 0:

            packets_to_move = random.randint(
                1,
                individual[path1]
            )

            individual[path1] -= packets_to_move
            individual[path2] += packets_to_move

    return individual


# ---------------------------------------------------------
# 8. GENETIC ALGORITHM
# ---------------------------------------------------------

def genetic_algorithm():

    population = create_population()

    best_solution = None
    best_fitness = 0

    for generation in range(GENERATIONS):

        new_population = []

        # Keep the best individual (elitism)
        current_best = max(population, key=fitness)

        if fitness(current_best) > best_fitness:

            best_fitness = fitness(current_best)
            best_solution = current_best.copy()

        new_population.append(current_best)

        # Create next generation
        while len(new_population) < POPULATION_SIZE:

            parent1 = selection(population)
            parent2 = selection(population)

            child1, child2 = crossover(parent1, parent2)

            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

        # Display progress
        if generation % 10 == 0:

            print(
                f"Generation {generation}: "
                f"Best Fitness = {best_fitness:.8f}"
            )

    return best_solution, best_fitness


# ---------------------------------------------------------
# 9. RUN THE GENETIC ALGORITHM
# ---------------------------------------------------------

best_solution, best_fitness = genetic_algorithm()


# ---------------------------------------------------------
# 10. OUTPUT BEST SOLUTION
# ---------------------------------------------------------

print("\n====================================")
print("OPTIMAL NETWORK PACKET DISTRIBUTION")
print("====================================")

for i in range(len(paths)):

    print(
        f"{paths[i]['name']}: "
        f"{best_solution[i]} packets"
    )

print("\nTotal Packets:", sum(best_solution))
print("Best Fitness:", best_fitness)