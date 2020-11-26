import random
import networkx
import numpy as np


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
        print(f"Coloring vertex {vertex}")
        vertex_color = 0
        is_possible = False
        color = 0
        while not is_possible:
            for i in range(self.n):
                if self.adj_matr[vertex][i] == 1 and color == self.colors[i]:
                    color += 1
                    break
                if i == self.n - 1:
                    is_possible = True
        print(f'Vertex {vertex} color is {color}')
        self.colors[vertex] = color
        for i in range(self.n):
            if self.adj_matr[vertex][i] == 1 and self.colors[i] == -1:
                self.color_vertex(i)

# seems like properly working algoritm
if __name__ == "__main__":
    graph = networkx.chvatal_graph()
    #graph = networkx.gnm_random_graph(7, 10)
    adj_matr = networkx.to_numpy_array(graph)
    greedy = Greedy(adj_matr)
    print(greedy.execute())