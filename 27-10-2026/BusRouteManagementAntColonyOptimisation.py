import random

# Bus network: stop -> {next_stop: travel_cost}
network = {
    'A': {'B': 4, 'C': 7},
    'B': {'A': 4, 'C': 2, 'D': 5},
    'C': {'A': 7, 'B': 2, 'D': 3, 'E': 6},
    'D': {'B': 5, 'C': 3, 'E': 2, 'F': 7},
    'E': {'C': 6, 'D': 2, 'F': 3},
    'F': {'D': 7, 'E': 3}
}

source, destination = 'A', 'F'

num_ants = 20
iterations = 100
alpha, beta = 1.0, 2.0
rho, Q = 0.5, 100
initial_pheromone = 0.1

# Initialize pheromone and heuristic matrices
pheromone = {
    u: {v: initial_pheromone for v in network[u]}
    for u in network
}

heuristic = {
    u: {v: 1 / cost for v, cost in network[u].items()}
    for u in network
}

best_route = None
best_cost = float('inf')

# ACO iterations
for iteration in range(iterations):
    all_routes = []

    # Each ant constructs a bus route
    for ant in range(num_ants):
        route = [source]
        visited = {source}
        current = source
        total_cost = 0

        while current != destination:
            candidates = [
                stop for stop in network[current]
                if stop not in visited
            ]

            if not candidates:
                break

            weights = [
                pheromone[current][stop] ** alpha
                * heuristic[current][stop] ** beta
                for stop in candidates
            ]

            next_stop = random.choices(
                candidates, weights=weights, k=1
            )[0]

            total_cost += network[current][next_stop]
            route.append(next_stop)
            visited.add(next_stop)
            current = next_stop

        # Accept only routes reaching the destination
        if current == destination:
            all_routes.append((route.copy(), total_cost))

            if total_cost < best_cost:
                best_cost = total_cost
                best_route = route.copy()

    # Evaporate pheromones
    for u in network:
        for v in network[u]:
            pheromone[u][v] *= (1 - rho)

    # Deposit pheromones on completed routes
    for route, cost in all_routes:
        deposit = Q / cost

        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            pheromone[u][v] += deposit
            pheromone[v][u] += deposit

# Display the best route
if best_route:
    print("Best Bus Route:", " -> ".join(best_route))
    print("Minimum Route Cost:", best_cost)
    print("Number of Bus Stops:", len(best_route))
else:
    print("No route found!")