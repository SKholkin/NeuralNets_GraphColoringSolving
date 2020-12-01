import random
import networkx
import numpy as np
import os
from ColoringDataSet import ColorData


class Greedy:
    def __init__(self, adj_matr, is_stochastic=False):
        self.adj_matr = adj_matr
        self.n = int(adj_matr.shape[0])
        self.colors = np.zeros(self.n, dtype=np.int32) - 1
        self.is_stochastic = is_stochastic

    def execute(self):
        if self.is_stochastic:
            first_vertex = random.randrange(self.n)
        else:
            first_vertex = 0
        self.color_vertex(first_vertex)
        return np.max(self.colors)

    def color_vertex(self, vertex):
        is_possible = False
        vertex_color = 0
        while not is_possible:
            for i in range(self.n):
                if self.adj_matr[vertex][i] == 1 and vertex_color == self.colors[i]:
                    vertex_color += 1
                    break
                if i == self.n - 1:
                    is_possible = True
        self.colors[vertex] = vertex_color
        for i in range(self.n):
            if self.adj_matr[vertex][i] == 1 and self.colors[i] == -1:
                self.color_vertex(i)


if __name__ == "__main__":
    color_data = ColorData(os.path.join('datasets', 'ColorData'))
    for graph_data in color_data:
        greedy = Greedy(graph_data[0])
        print(greedy.execute())
