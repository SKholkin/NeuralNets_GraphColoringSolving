from tqdm import tqdm
from copy import deepcopy
import sys
import os
import os.path as osp
import torch

from dataset_generator import write_instance, write_dataset_info


def main(folder_path):
    new_folder_name = folder_path + '_pt'
    mode = 'test' if 'test' in folder_path else 'train'
    max_vertices = 0
    min_vertices = 1000
    max_n_colors = 0
    for idx, item in tqdm(enumerate(os.listdir(folder_path))):
        with open(osp.join(folder_path, item), 'r') as f:
            lines = [item[:-1] for item in f.readlines()]
            begin_edge_list = 0
            end_edge_list = 0
            diff_edge_idx = 0
            chrom_number_idx =  0
            dimension_idx = 0
            for i, line in enumerate(lines):
                if line == 'EDGE_DATA_SECTION':
                    begin_edge_list = i
                if line == '-1':
                    end_edge_list = i
                if line == 'DIFF_EDGE':
                    diff_edge_idx = i + 1
                if line == 'CHROM_NUMBER':
                    chrom_number_idx = i + 1
                if 'DIMENSION' in line:
                    dimension_idx = i
            
            dimension = int(lines[dimension_idx].split()[1])
            tmp_adj_list = [[int(item) for item in line.split()] for line in lines[begin_edge_list + 1:end_edge_list]]
            base_adj_list = [[] for _ in range(dimension)]
            for out_vertex, in_vertex in tmp_adj_list:
                base_adj_list[out_vertex].append(in_vertex)
            diff_edge = [int(item) for item in lines[diff_edge_idx].split()]
            chrom_number = int(lines[chrom_number_idx].split()[0])

            adv_adj_list = deepcopy(base_adj_list)
            adv_adj_list[diff_edge[0]].append(diff_edge[1])
            adv_adj_list[diff_edge[1]].append(diff_edge[0])
            
            if not osp.exists(new_folder_name):
                os.mkdir(new_folder_name)
                os.mkdir(osp.join(new_folder_name, mode))
            elif not osp.exists(osp.join(new_folder_name, mode)):
                os.mkdir(osp.join(new_folder_name, mode))
            to_save_base = {'n_colors': chrom_number, 'is_solvable': True, 'adj_list': base_adj_list}
            torch.save(to_save_base, osp.join(new_folder_name, mode, f'graph_{2 * idx}.pt'))
            to_save_adv = {'n_colors': chrom_number, 'is_solvable': False, 'adj_list': adv_adj_list}
            torch.save(to_save_adv, osp.join(new_folder_name, mode, f'graph_{2 * idx + 1}.pt'))

            if dimension > max_vertices:
                max_vertices = dimension
            if dimension < min_vertices:
                min_vertices = dimension
            if chrom_number > max_n_colors:
                max_n_colors = chrom_number
    
    write_dataset_info(min_vertices, max_vertices, max_n_colors, mode, new_folder_name)

if __name__ == '__main__':
    main(sys.argv[1])
    