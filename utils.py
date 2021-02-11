
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


# working
def adj_list_to_adj_matr(adj_list):
    adj_matr = [([0 for j in range(len(adj_list))]) for i in adj_list]
    for i, vertex_info in enumerate(adj_list):
        for j in vertex_info:
            adj_matr[i][j] = 1
    return adj_matr


class AverageMetr:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return 0
