import numpy as np
from ortools.sat.python import cp_model


# ToDo: maybe test it more properly
def solve_by_csp(adj_matr, n_colors):
    model = cp_model.CpModel()
    model_vars = [model.NewIntVar(0, n_colors - 1, str(k)) for k, col in enumerate(adj_matr)]
    for i in range(len(adj_matr)):
        for j in range(i, len(adj_matr[i])):
            if adj_matr[i][j] == 1:
                model.Add(model_vars[i] != model_vars[j])
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:
        solution = [solver.Value(x) for x in model_vars]
        return solution
    else:
        return None


def basic_graph_gen(n, prob):
    # Erdős–Rényi <G,p> model
    adj_matr = np.random.choice([0, 1], size=[n, n], p=[1 - prob, prob])
    # make the matrix symmetric
    lower_ind = np.tril_indices(n, -1)
    adj_matr[lower_ind] = adj_matr.T[lower_ind]
    np.fill_diagonal(adj_matr, 0)
    return adj_matr


prob_by_color = {3: (0.05, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.25),
                        6: (0.25, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}


def basic_instance_gen(n, n_colors_min=3, n_colors_max=8):
    n_colors = np.random.randint(n_colors_min, n_colors_max)
    prob_of_edge = np.random.rand() * (prob_by_color[n_colors][1] - prob_by_color[n_colors][0]) + \
                   prob_by_color[n_colors][0]
    basic_graph = basic_graph_gen(n, prob_of_edge)
    solution = None
    n_colors = 1
    while solution is None:
        n_colors += 1
        solution = solve_by_csp(basic_graph, n_colors)
    return basic_graph, n_colors
