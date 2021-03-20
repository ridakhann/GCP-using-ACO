import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.function import nodes
import numpy as np
from numpy import random
import random


class Ant:
    def __init__(self, alpha, beta) -> None:
        self.g = None
        self.colors = {}
        self.start = None
        self.visited = []
        self.notvisited = []
        self.alpha = alpha
        self.beta = beta
        self.available_colors = []
        self.assigned_colors = {}
        self.colors_used = 0  # number of colours used in a solution

    def solution_init(self, g, colors, start=None):  # constructing solutions for each ant
        self.available_colors = sorted(colors.copy())
        keys = []
        for x in nodes:
            keys.append(x)
        for key in keys:
            self.assigned_colors[key] = None

        if start == None:  # if start not already assigned
            # randomly assigning a starting node/position for the ant
            self.start = random.choice(nodes)
        else:
            self.start = start

        self.visited = []
        self.notvisited = nodes.copy()

        if len(self.visited) == 0:
            # assigning the smallest possible colour to the vertex if it is the first step
            self.assigned_colors[self.start] = self.available_colors[0]
            self.visited.append(self.start)
            self.notvisited.remove(self.start)

        return self

    def color_graph(self):  # construction method, colouring graph
        tabu = []  # keep track of colours of neighbours to make sure not to colour the vertex same as its neighbor
        len_notvisited = len(self.notvisited)
        for i in range(len_notvisited):
            # applying the transition rule to choose the next node to colour
            next = self.choose_next()
            tabu = []  # emptying tabu list
            for j in range(number_of_nodes):
                # if neighbours add the colour of neighbour to tabu list
                if(adjacency_mat[next, j] == 1):
                    tabu.append(self.assigned_colors[j])
            for k in self.available_colors:  # checking list of over all colours available
                if (k not in tabu):
                    # assigning colour a colour that has not been restricted by the tabu list
                    self.assigned_colors[next] = k
                    self.visited.append(next)
                    self.notvisited.remove(next)
                    break
            # updating the total number of colours used
            self.colors_used = len(set(self.assigned_colors.values()))

    # calculating desiribility of a node based on the most number of conflicts a node has (degree of saturation)
    def desiribility(self, node=None):
        if node is None:  # case for the very first node
            node = self.start
        neighb = []  # list of neighbors
        for i in range(number_of_nodes):
            if adjacency_mat[node, i] == 1:
                neighb.append(self.assigned_colors[i])
        return len(set(neighb))  # unique colours within neighbours

    def get_phero(self, node, neighbor):  # returns the amount of pheromone on an edge
        return phero_mat[node, neighbor]

    def choose_next(self):  # transition method to choose next node to colour
        maxval = 0
        vals = []
        summation = 0
        candidates = []
        candidates_possible = []
        for j in self.notvisited:
            vals.append(self.get_phero(self.start, j)**self.alpha *
                        self.desiribility(j)**self.beta)
            candidates.append(j)
        # summation = sum(vals)
        # for v in range(len(vals)):
        #     vals[v] = vals[v]/summation
        maxval = max(vals)
        for i in range(len(candidates)):
            if vals[i] >= maxval:
                candidates_possible.append(candidates[i])
        candidate = random.choice(candidates_possible)
        self.start = candidate
        return candidate

    def get_phero_trail(self):  # gets pheromone trail of a path
        phero_trail = np.zeros((number_of_nodes+1, number_of_nodes+1), float)
        for i in nodes:
            for j in nodes:
                # if two nodes have the same colour
                if self.assigned_colors[i] == self.assigned_colors[j]:
                    phero_trail[i, j] = 1
        return phero_trail


def graph(path):  # constructing graph using txt file
    g = nx.Graph()
    with open(path, "r") as a_file:
        line = a_file.readline().split()
        for line in a_file:
            stripped_line = line.strip().split()
            g.add_edge(int(stripped_line[1]), int(stripped_line[2]))
    return g


def draw_graph(g, col_val):  # visualizing the graph and the final colours
    pos = nx.spring_layout(g)
    values = [col_val.get(node, 'green') for node in g.nodes()]
    nx.draw(g, pos, with_labels=True, node_color=values,
            edge_color='black', width=1, alpha=0.7)


def adjacency_matrix(g):  # constructing the adjacency matrix
    adj_matrix = np.zeros((number_of_nodes+1, number_of_nodes+1), int)
    for node in nodes:
        for adj_node in g.neighbors(node):
            adj_matrix[node, adj_node] = 1
    return adj_matrix


def init_pheros(g):  # initialising pheromone matrix
    phero_mat = np.ones((number_of_nodes+1, number_of_nodes+1), float)
    for node in g:
        for neighbor in g.neighbors(node):
            # if nodes are neighbours, pheremone value = 0
            phero_mat[node][neighbor] = 0
    return phero_mat


def colony(alpha, beta):  # initilizing colony of ants
    ants = []
    for i in range(number_of_ants):
        a = Ant(alpha, beta)
        ants.append(a.solution_init(g, colors))
    return ants


def evaporation():  # applying evaporation on the pheremone matrix
    for node in nodes:
        for neighbor in nodes:
            phero_mat[node, neighbor] = phero_mat[node, neighbor]*(1-rho)


def getbest():  # returns the best solution yet
    global phero_mat
    best = 0  # least number of colours used
    elite = None  # best solution
    for ant in ants:
        if best == 0:
            best = ant.colors_used
            elite = ant
        elif ant.colors_used < best:  # if a new best appears
            best = ant.colors_used
            elite = ant
    # update pheromones matrix
    # get pheremone trail of the best solution
    elite_phero = elite.get_phero_trail()
    phero_mat = phero_mat + elite_phero  # pheromone intensification according to best sol so far
    return elite.colors_used, elite.assigned_colors


def algo(gr, numberants, iterations, alpha, beta, evap):
    global number_of_nodes
    global number_of_edges
    global nodes
    global number_of_ants
    global adjacency_mat
    global phero_mat
    global colors
    global ants
    global g

    number_of_ants = numberants

    ants = []
    colors = []
    g = gr

    solution = {}
    colors_in_final = 0
    iterations_needed = 0

    number_of_nodes = nx.number_of_nodes(gr)
    nodes = []
    for node in g.nodes():
        nodes.append(node)
    nodes.sort()

    adjacency_mat = adjacency_matrix(gr)
    phero_mat = init_pheros(gr)
    number_of_nodes = nx.number_of_nodes(gr)
    for i in range(number_of_nodes):
        colors.append(i)
    aver=[]
    temp = []
    best=[]

    for i in range(iterations):
        ants = colony(alpha, beta)
        for ant in ants:
            ant.color_graph()
        evaporation()
        best_colors, best_sol = getbest()
        temp.append(best_colors)
        if(colors_in_final == 0):
            colors_in_final = best_colors
            solution = best_sol
            iterations_needed = i+1
            # aver.append(best_colors)
        elif(best_colors < colors_in_final):  # a new better solution appears swap it with current best
            colors_in_final = best_colors
            solution = best_sol
            iterations_needed = i+1
        best.append(colors_in_final)
    # summ=sum(best)
    # aver.append(best[0])
    for i in range(len(temp)):
        aver.append(sum(temp[:i+1])/(i+1))

    return colors_in_final, solution, iterations_needed, best, aver
g = graph('gcol1.txt')
number_of_nodes = 0
nodes = []
number_of_ants = 0
alpha =0.9
beta=0.5
rho = 0.8
numIterations=5
phero_mat = np.ones((number_of_nodes, number_of_nodes), float)
adjacency_mat = np.zeros((number_of_nodes, number_of_nodes), float)
final_costs, final_solution, iterations_needed, best, aver = algo(g, 20, numIterations, alpha, beta, 0.8)
print(final_costs, final_solution, iterations_needed)
draw_graph(g, final_solution)
plt.show()

iterations = range(numIterations)
plt.plot(iterations, aver)
plt.plot(iterations, best)
plt.legend(["Average Fitness", "Best Fitness"])
plt.title('Fitness Vs Iterations')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.show()
