import numpy as np
from utils import print_weight_matrix
from ortools.sat.python import cp_model


def basic_graph_gen(n, prob):
    # Erdős–Rényi <G,p> model
    adj_matr = np.random.choice([0, 1], size=[n, n], p=[1 - prob, prob])
    # make the matrix symmetric
    lower_ind = np.tril_indices(n, -1)
    adj_matr[lower_ind] = adj_matr.T[lower_ind]
    np.fill_diagonal(adj_matr, 0)
    return adj_matr


def solve_by_csp(adj_matr, n_colors):
    model = cp_model.CpModel()
    model_vars = [model.NewIntVar(0, n_colors - 1, str(k)) for k, col in enumerate(adj_matr)]
    for i in range(len(adj_matr)):
        for j in range(i, len(adj_matr[i])):
            if adj_matr[i][j] == 1:
                model.Add(model_vars[i] != model_vars[j])
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    # ToDo: deal with status
    for i in range(0, len(adj_matr)):
        print(f'{i} = {solver.Value(model_vars[i])} ')
    print(status)


if __name__ == '__main__':
    n = 10
    adj_matr = basic_graph_gen(n, 0.4)
    print_weight_matrix(adj_matr)
    solve_by_csp(adj_matr, n_colors=3)
