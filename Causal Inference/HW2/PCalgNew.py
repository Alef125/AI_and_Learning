import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm


# Information of all partial correlations
class GraphCorrelations:
    def __init__(self, corrs):
        self.corrs = corrs

    def calculate_correlation(self, i, j, dep_set):
        if not dep_set:
            return self.corrs[i][j]
        else:
            h = dep_set[0]
            reduced_set = list(set(dep_set) - {h})  # list(set(dep_set.variables) - set(h))
            prev_corr = self.calculate_correlation(i, j, reduced_set)
            i_corr = self.calculate_correlation(i, h, reduced_set)
            j_corr = self.calculate_correlation(j, h, reduced_set)
            corr = (prev_corr - i_corr * j_corr) / \
                np.sqrt((1 - np.power(i_corr, 2)) * (1 - np.power(j_corr, 2)))
            return corr


def z_trans(rho):
    return 0.5 * np.log((1 + rho) / (1 - rho))


def adjacent_sets(i, j, l_i, adj, nodes):
    # Could Throw Exception in: j is adjacent with i
    if l_i == 0:
        return [()]
    neighbours = []
    for ii in range(nodes):
        if adj[i, ii]:
            neighbours.append(ii)
    neighbours = list(set(neighbours) - {j})
    return list(itertools.combinations(neighbours, l_i))


def all_pairs(adj, l_i, nodes):
    pairs = []
    for i in range(nodes):
        neighbours_len = 0
        for ii in range(nodes):
            if adj[i, ii]:
                neighbours_len = neighbours_len + 1
        i_neighbour_size = neighbours_len
        if i_neighbour_size >= l_i + 1:
            for j in range(nodes):
                if adj[i, j]:
                    pairs.append([i, j])
    return pairs


n = 200  # Number of Nodes(Vertices)
file_name = 'pc-data.csv'
file_data = pd.read_csv(file_name, sep='\t', header=None)
v = (np.array(file_data.values)).T  # Read data of nodes
data_size = v[0].size  # = 5000
accept_level = 2.3263  # t_n > Q^-1(alpha/2) = Q^-1(0.01) = 2.3263
adjacency = np.full(shape=(n, n), fill_value=True)
for cnt in range(n):
    adjacency[cnt, cnt] = 0

# Initializing with zero order tests
init_correlations = np.zeros(shape=(n, n))
for row in range(n):
    for col in range(row, n):  # corr(xi, xi) is calculated and it's corrcoeff = 1
        test = np.corrcoef(v[row], v[col])[1, 0]
        init_correlations[row][col] = test
        if row != col:
            init_correlations[col][row] = test

g = GraphCorrelations(init_correlations)

normal_level = accept_level / np.sqrt(data_size - 3)

# Start
for l in range(n):
    # new pair (v1,v2), v1 and v2 are adjacent, adj(v1)\v2 >= l
    v_s = all_pairs(adjacency, l, n)
    print "l=", l
    for pair in tqdm(v_s):
        v1 = pair[0]
        v2 = pair[1]
        v1_adjacent = adjacent_sets(v1, v2, l, adjacency, n)
        for kk in v1_adjacent:
            k = list(kk)
            corr_coef = g.calculate_correlation(v1, v2, k)
            z_rho = z_trans(corr_coef)
            if np.abs(z_rho) < normal_level:
                adjacency[v1][v2] = False
                adjacency[v2][v1] = False
                break

print "Finally, DAG Edges Are:"
for row in range(n):
    for col in range(row + 1, n):
        if adjacency[row][col]:
            print "(", row, ",", col, ")"
# np.savetxt("adjacency.txt",  np.int32(adjacency), fmt="%1.0d", delimiter=' ', newline='\n')
np.savetxt("adjacent_pc.csv", np.int32(adjacency), delimiter=",", fmt="%1.1d")
