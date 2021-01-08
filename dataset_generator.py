import numpy as np
from graph_generator import basic_graph_gen
from ortools.sat.python import cp_model


def get_edges(adj_matr, is_edge=True):
    # if is_edge = False then choose not_edges
    is_edge = 1 if is_edge else 0
    not_edges = []
    for i in range(len(adj_matr)):
        for j in range(len(adj_matr)):
            if adj_matr[i][j] == 0:
                not_edges.append((i, j))
    return not_edges


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
        print('Solving was successful')
        return solution
    else:
        print('Solving was NOT successful')
        return None


def write_graph(graph, path):
    pass


def sort_edges_by_vertex_ranking(edges, adj_matr):
    pass


def create_adversarial_graph_my_version(basic_graph, n_colors):
    solution = solve_by_csp(basic_graph, n_colors)
    if solution is not None:
        not_edges = get_edges(basic_graph, is_edge=False)
        # ToDo: sort edges by vertex ranking
        new_graph = basic_graph.copy()
        for k, (i, j) in enumerate(not_edges):
            new_graph[i][j] = 1
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is None:
                return new_graph
    else:
        edges = get_edges(basic_graph, is_edge=False)
        # ToDo: sort edges by vertex ranking
        new_graph = basic_graph.copy()
        for k, (i, j) in enumerate(edges):
            new_graph[i][j] = 0
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is not None:
                return new_graph
    # actually algo won't get this far but just to be sure...
    return None


def create_adversarial_graph_gnn_gcp(basic_graph, n_colors):
    solution = solve_by_csp(basic_graph, n_colors)
    if solution is not None:
        not_edges = get_edges(basic_graph, is_edge=False)
        # ToDo: sort edges by vertex ranking
        for k, (i, j) in enumerate(not_edges):
            new_graph = basic_graph.copy()
            new_graph[i][j] = 1
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is None:
                return new_graph
        if k == len(not_edges):
            print("Couldn't find diff edge")
            return None
    else:
        edges = get_edges(basic_graph, is_edge=True)
        for k, (i, j) in enumerate(edges):
            new_graph = basic_graph.copy()
            new_graph[i][j] = 0
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is not None:
                return new_graph
        if k == len(edges):
            print("Couldn't find diff edge")
            return None


def generate_dataset_gnn_gcp(nmin, nmax, samples, path):
    prob_by_color = {3: (0.01, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.3),
                        6: (0.2, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}
    for iter in range(samples):
        n_colors = np.random.randint(3, 8)
        n = np.random.randint(nmin, nmax)
        prob_of_edge = np.random.rand() * (prob_by_color[n_colors][1] - prob_by_color[n_colors][0]) + prob_by_color[n_colors][0]
        basic_graph = basic_graph_gen(n, prob_of_edge)
        solution = solve_by_csp(basic_graph, n_colors)
        adversarial_graph = create_adversarial_graph_gnn_gcp()
        write_graph(basic_graph, path)
        if adversarial_graph is not None:
            write_graph(adversarial_graph, path)
