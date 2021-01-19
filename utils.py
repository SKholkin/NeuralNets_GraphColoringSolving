
def print_weight_matrix(weight_matrix):
    print('Weight Matrix')
    for i in range(len(weight_matrix)):
        print(weight_matrix[i])


def adj_matr_to_adj_list(adj_matr):
    adj_list = [[] for i in range(len(adj_matr))]
    adj_matr = adj_matr.copy()
    for i in range(len(adj_matr)):
        for j in range(len(adj_matr)):
            if adj_matr[i][j] == 1:
                adj_list[i].append(j)
                adj_list[j].append(i)
                adj_matr[i][j] = 0
                adj_matr[j][i] = 0
    return adj_list
