import numpy as np
from utils import print_weight_matrix


def basic_graph_gen(n, prob):
    # Erdős–Rényi <G,p> model
    adj_matr = np.random.choice([0, 1], size=[n, n], p=[1 - prob, prob])
    # make the matrix symmetric
    lower_ind = np.tril_indices(n, -1)
    adj_matr[lower_ind] = adj_matr.T[lower_ind]
    np.fill_diagonal(adj_matr, 0)
    return adj_matr


#if __name__ == '__main__':
#    n = 10
#    adj_matr = basic_graph_gen(n, 0.4)
#    print_weight_matrix(adj_matr)
#    solve_by_csp(adj_matr, n_colors=3)
