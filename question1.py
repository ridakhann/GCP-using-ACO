import networkx as nx
import matplotlib.pyplot as plt
from networkx.classes.function import nodes
import numpy as np
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

    def solution_init(self, g, colors, start=None):
        self.available_colors = sorted(colors.copy())
        keys = [x for x in nodes]
        for key in keys:
            self.assigned_colors[key] = None

        if start == None:
            self.start = random.choice(nodes)
        else:
            self.start = start

        self.visited = []
        self.notvisited = nodes.copy()

        if len(self.visited) == 0:
            self.assign_color(self.start, self.available_colors[0])
        return self

    def assign_color(self, node, color):
        self.assigned_colors[node] = color
        self.visited.append(node)
        self.notvisited.remove(node)

    def color_graph(self):
        tabu = []
        len_notvisited = len(self.notvisited)
        for i in range(len_notvisited):
            next = self.choose_next()
            tabu = []  # emptying tabu list
            for j in range(number_of_nodes):
                if(adjacency_mat[next, j] == 1):
                    tabu.append(self.assigned_colors[j])
            for k in self.available_colors:
                if (k not in tabu):
                    self.assign_color(next, k)
                    break
            self.colors_used = len(set(self.assigned_colors.values()))

    def diff_color(self, node=None): #desiribility
        if node is None:
            node = self.start
        neighb = []
        for i in range(number_of_nodes):
            if adjacency_mat[node, i] == 1:
                neighb.append(self.assigned_colors[i])
        return len(set(neighb)) #unique colours within neighbours

    def get_phero(self, node, neighbor):
        return phero_mat[node, neighbor]

    def choose_next(self):  # check this later
        maxval = 0
        vals = []
        summation = 0
        candidates = []
        candidates_possible = []
        for j in self.notvisited:
            vals.append((self.get_phero(self.start, j) ** self.alpha)
                        * (self.diff_color(j)**self.beta))
            candidates.append(j)
        summation = sum(vals)
        # print(summation)
        for v in vals:
            v = v/summation
        print(vals)
        maxval = max(vals)
        for i in range(len(candidates)):
            if vals[i] >= maxval:
                candidates_possible.append(candidates[i])
        candidate = random.choice(candidates_possible)
        self.start = candidate
        return candidate

    def get_phero_trail(self):
        phero_trail = np.zeros((number_of_nodes+1, number_of_nodes+1), float)
        for i in nodes:
            for j in nodes:
                if self.assigned_colors[i] == self.assigned_colors[j]:
                    phero_trail[i, j] = 1
        return phero_trail


def graph(path):
    g = nx.Graph()
    with open(path, "r") as a_file:
        line = a_file.readline().split()
        number_of_nodes = line[-2]
        number_of_edges = line[-1]
        for line in a_file:
            stripped_line = line.strip().split()
            # print(stripped_line)
            g.add_edge(int(stripped_line[1]), int(stripped_line[2]))
    return g


def draw_graph(g, col_val):
    pos = nx.spring_layout(g)
    values = [col_val.get(node, 'blue') for node in g.nodes()]
    # with_labels=true is to show the node number in the output graph
    nx.draw(g, pos, with_labels=True, node_color=values,
            edge_color='black', width=1, alpha=0.7)


def init_colors(g):
    colors = []
    grundy = len(nx.degree_histogram(g))
    for c in range(grundy):
        colors.append(c)
    # print(grundy, len(g.nodes()))
    return colors


def adjacency_matrix(g):
    adj_matrix = np.zeros((number_of_nodes+1, number_of_nodes+1), int)
    for node in nodes:
        for adj_node in g.neighbors(node):
            adj_matrix[node, adj_node] = 1
    return adj_matrix


def init_pheros(g):
    phero_mat = np.ones((number_of_nodes+1, number_of_nodes+1), float)
    for node in g:
        for neighbor in g.neighbors(node):
            phero_mat[node][neighbor] = 0
    return phero_mat


def colony():
    ants = []
    for i in range(number_of_ants):
        a = Ant(0.8, 0.8)
        ants.append(a.solution_init(g, colors))
    return ants


def evaporation():
    for node in nodes:
        for neighbor in nodes:
            phero_mat[node, neighbor] = phero_mat[node, neighbor]*(1-rho)


def getbest():  # check this
    global phero_mat
    best = 0
    elite = None
    for ant in ants:
        if(best == 0):
            best = ant.colors_used
            elite = ant
        elif ant.colors_used < best:
            best = ant.colors_used
            elite = ant
    # update pheromones mat
    elite_phero = elite.get_phero_trail()
    phero_mat = phero_mat + elite_phero
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
    colors = init_colors(gr)
    phero_mat = init_pheros(gr)
    number_of_nodes = nx.number_of_nodes(gr)

    for i in range(iterations):
        ants = colony()
        for ant in ants:
            ant.color_graph()
        evaporation()
        best_colors, best_sol = getbest()
        if(colors_in_final == 0):
            colors_in_final = best_colors
            solution = best_sol
            iterations_needed = i+1
        elif(best_colors < colors_in_final):
            colors_in_final = best_colors
            solution = best_sol
            iterations_needed = i+1
    return colors_in_final, solution, iterations_needed


g = graph('t.txt')
number_of_nodes = 0
nodes = []
number_of_ants = 0
rho = 0.8
phero_mat = np.ones((number_of_nodes, number_of_nodes), float)
adjacency_mat = np.zeros((number_of_nodes, number_of_nodes), float)
final_costs, final_solution, iterations_needed = algo(g, 20, 5, 0.8, 0.8, 0.8)
print(final_costs, final_solution, iterations_needed)
draw_graph(g, final_solution)
plt.show()
