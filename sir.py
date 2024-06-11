import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random

num_nodes = 100
initial_infected = 5
beta = 0.3
gamma = 0.1
steps = 50

G = nx.erdos_renyi_graph(num_nodes, 0.1)

for node in G.nodes:
    G.nodes[node]['state'] = 'S'

initial_infected_nodes = random.sample(list(G.nodes), initial_infected)
for node in initial_infected_nodes:
    G.nodes[node]['state'] = 'I'

def update_node_states(G, beta, gamma):
    new_states = {}
    for node in G.nodes:
        if G.nodes[node]['state'] == 'S':
            infected_neighbors = [n for n in G.neighbors(node) if G.nodes[n]['state'] == 'I']
            if infected_neighbors and random.random() < beta:
                new_states[node] = 'I'
        elif G.nodes[node]['state'] == 'I':
            if random.random() < gamma:
                new_states[node] = 'R'
    for node, state in new_states.items():
        G.nodes[node]['state'] = state

def count_states(G):
    state_counts = {'S': 0, 'I': 0, 'R': 0}
    for node in G.nodes:
        state_counts[G.nodes[node]['state']] += 1
    return state_counts

S, I, R = [], [], []

for step in range(steps):
    state_counts = count_states(G)
    S.append(state_counts['S'])
    I.append(state_counts['I'])
    R.append(state_counts['R'])
    update_node_states(G, beta, gamma)

plt.figure(figsize=(8, 4))
plt.plot(range(steps), S, label='Susceptible')
plt.plot(range(steps), I, label='Infected')
plt.plot(range(steps), R, label='Recovered')
plt.xlabel('Time steps')
plt.ylabel('Number of individuals')
plt.title('SIR Model on a Network')
plt.legend()
plt.grid(True)
plt.show()
