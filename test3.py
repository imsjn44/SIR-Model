import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'nepal_covid.csv'
df = pd.read_csv(file_path)

# Assume the total population is a constant value
total_population = 288300000

# Calculate the Susceptible population
df['Susceptible'] = total_population - (df['Infected'] + df['Recovered'] + df['Deaths'])

# Normalize the "Susceptible", "Infected", and "Recovered" columns to 250
normalized_population = 250
columns_to_normalize = ['Susceptible', 'Infected', 'Recovered']
df[columns_to_normalize] = df[columns_to_normalize].div(total_population) * normalized_population

# Rounding off the possible floating values
df[columns_to_normalize] = df[columns_to_normalize].round()

# Verify the normalization
print(df[columns_to_normalize].sum(axis=1).head())  # Each row should sum to 300

# Calculate the total sum of S, I, R columns
total_susceptible = int(df['Susceptible'].sum())
total_infected = int(df['Infected'].sum())
total_recovered = int(df['Recovered'].sum())

# Calculate beta and gamma
def calculate_beta_gamma(df):
    new_infections = df['Infected'].diff().fillna(0).clip(lower=0)  # diff between two consecutive rows of infected
    new_recoveries = df['Recovered'].diff().fillna(0).clip(lower=0)  # diff between two consecutive rows of recovered
    susceptible = df['Susceptible']
    infected = df['Infected']
    
    # Calculate beta
    beta_values = (new_infections / (susceptible * infected)).replace([np.inf, -np.inf], np.nan).dropna()
    beta = beta_values.mean()
    
    # Calculate gamma
    gamma_values = (new_recoveries / infected).replace([np.inf, -np.inf], np.nan).dropna()
    gamma = gamma_values.mean()
    
    return beta, gamma

beta, gamma = calculate_beta_gamma(df)
print(f"Calculated beta: {beta:.6f}")
print(f"Calculated gamma: {gamma:.6f}")

# Function to initialize the SIR model
def initialize_SIR(graph, total_infected, total_recovered):
    # Initialize node attributes
    for node in graph.nodes:
        graph.nodes[node]['state'] = 'S'  # All nodes are susceptible initially

    # Randomly select initial infected nodes
    infected_nodes = np.random.choice(graph.nodes, total_infected, replace=False)  # selection of nodes without replacing the previous node
    for node in infected_nodes:
        graph.nodes[node]['state'] = 'I'  # Set the state of infected nodes to 'I'
    
    # Randomly select recovered nodes from the remaining susceptible nodes
    remaining_nodes = [node for node in graph.nodes if graph.nodes[node]['state'] == 'S']
    recovered_nodes = np.random.choice(remaining_nodes, total_recovered, replace=False)
    for node in recovered_nodes:
        graph.nodes[node]['state'] = 'R'  # Set the state of recovered nodes to 'R'

# Function to run a single step of the SIR model
def sir_step(graph, beta, gamma):
    new_infected = [] #list of newly added infected values
    new_recovered = []#list of newly added recovered values

    # Iterate over all nodes
    for node in graph.nodes:
        if graph.nodes[node]['state'] == 'I':
            # Infect susceptible neighbors with probability beta
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor]['state'] == 'S' and np.random.rand() < beta:
                    new_infected.append(neighbor)

            # Recover with probability gamma
            if np.random.rand() < gamma:
                new_recovered.append(node)

    # Update node states
    for node in new_infected:
        graph.nodes[node]['state'] = 'I'
    for node in new_recovered:
        graph.nodes[node]['state'] = 'R'

# Function to plot the SIR model with beta and gamma
def plot_SIR(graph, pos, beta, gamma, step):
    colors = {'S': 'yellow', 'I': 'red', 'R': 'green'}
    node_colors = [colors[graph.nodes[node]['state']] for node in graph.nodes]
    plt.figure(figsize=(10, 10))
    nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=400, font_size=10)
    plt.title(f"SIR Model - Step {step}\nβ = {beta:.6f}, γ = {gamma:.6f}")
    plt.show()

# Parameters
N = normalized_population  # Number of nodes (equal to the normalized total population)

# Create a graph
graph = nx.erdos_renyi_graph(N, 0.1)

# Initialize the SIR model
initialize_SIR(graph, total_infected, total_recovered)

# Calculate positions once and reuse them
pos = nx.spring_layout(graph)

# Plot the initial state of the network
plot_SIR(graph, pos, beta, gamma, step=0)

# Run the SIR model for 20 steps
for step in range(1, 20):
    sir_step(graph, beta, gamma)
    plot_SIR(graph, pos, beta, gamma, step)
