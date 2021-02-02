import numpy as np
import os
from graph_generator import basic_graph_gen
from torch import save
import logging
import datetime
from utils import print_weight_matrix, adj_matr_to_adj_list
from ColorDataset import prepare_folders, ColorDataset
from graph_generator import solve_by_csp, basic_instance_gen
import random
import argparse


a = str.maketrans('- :.', '____')
logging.basicConfig(filename=f"log/{str(datetime.datetime.today()).translate(str.maketrans('- :.', '____'))}.log",
                    filemode='w')


def get_edges(adj_matr, is_edge=True):
    # if is_edge = False then choose not_edges
    is_edge = 1 if is_edge else 0
    not_edges = []
    for i in range(len(adj_matr)):
        for j in range(i + 1, len(adj_matr)):
            if adj_matr[i][j] == is_edge:
                not_edges.append((i, j))
    return not_edges


def write_instance(graph, n_colors, path):
    to_save = [n_colors]
    to_save += adj_matr_to_adj_list(graph)
    save(to_save, path)


def sort_edges_by_vertex_ranking(edges, adj_matr):
    ranks_by_vertices = [sum(vertex) for vertex in adj_matr]
    sorted_edges = sorted(edges, key=lambda edge: max(ranks_by_vertices[edge[0]], ranks_by_vertices[edge[1]]),
                          reverse=True)
    return sorted_edges


def random_sort_edges(edges):
    random.shuffle(edges)
    return edges


def create_adversarial_graph_my_version(basic_graph, n_colors):
    solution = solve_by_csp(basic_graph, n_colors)
    if solution is not None:
        not_edges = get_edges(basic_graph, is_edge=False)
        sort_edges_by_vertex_ranking(not_edges, basic_graph)
        not_edges = random_sort_edges(not_edges)
        new_graph = basic_graph.copy()
        for k, (i, j) in enumerate(not_edges):
            new_graph[i][j] = 1
            new_graph[j][i] = 1
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is None:
                while new_solution is None:
                    n_colors += 1
                    new_solution = solve_by_csp(new_graph, n_colors)
                return new_graph, n_colors
    else:
        edges = get_edges(basic_graph, is_edge=True)
        edges = random_sort_edges(edges)
        new_graph = basic_graph.copy()
        for k, (i, j) in enumerate(edges):
            new_graph[i][j] = 0
            new_graph[j][i] = 0
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is not None:
                return new_graph, n_colors
    # actually algo won't get this far but just to be sure...
    return None


def create_adversarial_graph_gnn_gcp(basic_graph, n_colors):
    solution = solve_by_csp(basic_graph, n_colors)
    if solution is not None:
        not_edges = get_edges(basic_graph, is_edge=False)
        not_edges = sort_edges_by_vertex_ranking(not_edges, basic_graph)
        for k, (i, j) in enumerate(not_edges):
            new_graph = basic_graph.copy()
            new_graph[i][j] = 1
            new_graph[j][i] = 1
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is None:
                return new_graph
        print("Couldn't find diff edge")
        return None
    else:
        edges = get_edges(basic_graph, is_edge=True)
        edges = sort_edges_by_vertex_ranking(edges, basic_graph)
        for k, (i, j) in enumerate(edges):
            new_graph = basic_graph.copy()
            new_graph[i][j] = 0
            new_graph[j][i] = 0
            new_solution = solve_by_csp(new_graph, n_colors)
            if new_solution is not None:
                return new_graph
        print("Couldn't find diff edge")
        return None


def generate_dataset_gnn_gcp(nmin, nmax, samples, root):
    prepare_folders(root)
    prob_by_color = {3: (0.05, 0.1), 4: (0.1, 0.2), 5: (0.2, 0.25),
                        6: (0.25, 0.3), 7: (0.3, 0.4), 8: (0.4, 0.5)}
    for iter in range(samples):
        n_colors = np.random.randint(3, 8)
        n = np.random.randint(nmin, nmax)
        prob_of_edge = np.random.rand() * (prob_by_color[n_colors][1] - prob_by_color[n_colors][0]) + prob_by_color[n_colors][0]
        basic_graph = basic_graph_gen(n, prob_of_edge)
        basic_graph, n_colors = basic_instance_gen(n)
        adversarial_graph, n_adv_colors = create_adversarial_graph_my_version(basic_graph, n_colors)
        write_instance(basic_graph, n_colors, os.path.join(root, 'ColorDataset', 'basic', f'graph_{iter}.pt'))
        if adversarial_graph is not None:
            print('Found an adversarial graph')
            write_instance(adversarial_graph, n_adv_colors, os.path.join(root, 'ColorDataset', 'adv', f'graph_{iter}.pt'))
        else:
            print('Not found an adversarial graph')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, help='Number of dataset samples default=1000', default=1000)
    parser.add_argument('--nmin', type=int, help='Minimal number of vertices in dataset', default=10)
    parser.add_argument('--nmax', type=int, help='Maximum number of vertices in dataset', default=20)
    parser.add_argument('--root', type=str, help='Dataset root path')
    args = parser.parse_args()
    generate_dataset_gnn_gcp(nmin=args.nmin, nmax=args.nmax, samples=args.samples, root=args.root)
    dataset = ColorDataset('datasets')
