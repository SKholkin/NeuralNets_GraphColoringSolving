from abc import ABC

from torch_geometric.data import InMemoryDataset, Dataset
import os.path as osp
import os
import torch


class ColorBenchmark(Dataset, ABC):

    def __init__(self, root):
        self.dataset_abs_path = osp.join(os.path.dirname(os.path.abspath(__file__)), root)
        super(ColorBenchmark, self).__init__(root)
        self.data = []
        for idx in range(len(self.processed_file_names)):
            data_loaded = torch.load(osp.join('datasets', 'ColorBenchmark', 'processed', self.processed_file_names[idx]))
            vertices = int(data_loaded['vertices'])
            for idx in range(len(data_loaded['edges'][0])):
                data_loaded['edges'][0][idx] = int(data_loaded['edges'][0][idx])
                data_loaded['edges'][1][idx] = int(data_loaded['edges'][1][idx])
            edges = torch.Tensor(data_loaded['edges']).type(torch.int32)
            self.data.append((vertices, edges))

    def __getitem__(self, idx):
        vertices, edges = self.data[idx]
        adj = torch.zeros((vertices, vertices), dtype=torch.int32)
        for edges_idx in range(edges.shape[1]):
            adj[edges[0][edges_idx]][edges[1][edges_idx]] = 1
            adj[edges[1][edges_idx]][edges[0][edges_idx]] = 1
        colors = vertices
        vert_to_color = torch.zeros((vertices, colors), dtype=torch.int32)
        is_solvable = True
        numb_of_edges = edges.shape[1]
        return adj, colors, vert_to_color, is_solvable, vertices, numb_of_edges

    @property
    def raw_file_names(self):
        with open(osp.join(self.dataset_abs_path, 'raw', 'layout.txt')) as raw_layout:
            return [line.replace('\n', '') for line in raw_layout.readlines()]

    @property
    def processed_file_names(self):
        with open(osp.join(self.dataset_abs_path, 'processed', 'layout.txt')) as processed_layout:
            return [line.replace('\n', '') for line in processed_layout.readlines()]

    def download(self):
        raise NotImplementedError('Link will be provided later')

    def process(self):
        datalist = []
        for raw_file_name, processed_file_name in zip(self.raw_file_names, self.processed_file_names):
            edges = []
            data_from_raw = []
            with open(osp.join('datasets', 'ColorBenchmark', 'raw', raw_file_name), 'r+') as graph:
                for line in graph.readlines():
                    if not (line[0] == 'c' or line[0] == '\n'):
                        data_from_raw.append(line)

            with open(osp.join('datasets', 'ColorBenchmark', 'raw', raw_file_name), 'r+') as graph:
                graph.truncate()
                graph.writelines(data_from_raw)

            with open(osp.join('datasets', 'ColorBenchmark', 'raw', raw_file_name), 'r+') as graph:
                main_info = graph.readline()
                edges.extend(graph.readlines())
                in_edges = [str(int(pair.split(' ')[1]) - 1) for pair in edges]
                out_edges = [str(int(pair.split(' ')[2]) - 1) for pair in edges]

            datalist.append({'vertices': main_info.split(' ')[2], 'edges': (in_edges, out_edges)})
            torch.save(datalist[-1], osp.join('datasets', 'ColorBenchmark', 'processed', processed_file_name))
