import numpy as np
import itertools


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


def show_graph(adj, nodes):
    for r in range(nodes):
        print adj[r][:]


def pc(n, data, accept_level):
    # v = (np.array(data.values)).T  # Read data of nodes
    v = data
    data_size = v[:, 0].size
    adjacency = np.full(shape=(n, n), fill_value=True)
    for cnt in range(n):
        adjacency[cnt, cnt] = False

    # Initializing with zero order tests
    init_correlations = np.zeros(shape=(n, n))
    for row in range(n):
        init_correlations[row][row] = 1
        for col in range(row + 1, n):  # corr(xi, xi) is calculated and it's corrcoeff = 1
            test = np.corrcoef(v[:, row], v[:, col])[1, 0]
            init_correlations[row][col] = test
            if row != col:
                init_correlations[col][row] = test
    g = GraphCorrelations(init_correlations)

    normal_level = accept_level / np.sqrt(data_size - 3)

    # Start
    for l in range(n):
        # new pair (v1,v2), v1 and v2 are adjacent, adj(v1)\v2 >= l
        v_s = all_pairs(adjacency, l, n)
        for pair in v_s:
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
    return adjacency


def pc_stable(n, data, accept_level):
    v = data
    # v = (np.array(data.values)).T  # Read data of nodes
    data_size = v[:, 0].size
    adjacency = np.full(shape=(n, n), fill_value=True)
    for cnt in range(n):
        adjacency[cnt, cnt] = False

    # Initializing with zero order tests
    init_correlations = np.zeros(shape=(n, n))
    for row in range(n):
        init_correlations[row][row] = 1
        for col in range(row + 1, n):  # corr(xi, xi) is calculated and it's corrcoeff = 1
            test = np.corrcoef(v[:, row], v[:, col])[1, 0]
            init_correlations[row][col] = test
            if row != col:
                init_correlations[col][row] = test

    g = GraphCorrelations(init_correlations)

    normal_level = accept_level / np.sqrt(data_size - 3)

    # Start
    for l in range(n):
        # new pair (v1,v2), v1 and v2 are adjacent, adj(v1)\v2 >= l
        v_s = all_pairs(adjacency, l, n)
        cut_pairs = []
        for pair in v_s:
            v1 = pair[0]
            v2 = pair[1]
            v1_adjacent = adjacent_sets(v1, v2, l, adjacency, n)
            for kk in v1_adjacent:
                k = list(kk)
                corr_coef = g.calculate_correlation(v1, v2, k)
                z_rho = z_trans(corr_coef)
                if corr_coef >= 1 or corr_coef <= -1:
                    print(corr_coef)
                if np.abs(z_rho) < normal_level:
                    cut_pairs.append([v1, v2])
                    # adjacency[v1][v2] = 0
                    # adjacency[v2][v1] = 0
                    break
        for cutting in cut_pairs:
            adjacency[cutting[0]][cutting[1]] = False
            adjacency[cutting[1]][cutting[0]] = False
    return adjacency
