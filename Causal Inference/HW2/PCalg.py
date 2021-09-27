import numpy as np
import itertools
import pandas as pd
from tqdm import tqdm


# Assigns a correlation to a set
class PartialCorrelation:
    def __init__(self, s, corr):
        self.on_set = s
        self.corr = corr


# For each pairs, saves the correlation tests
class CorrelationTests:
    def __init__(self):
        self.tests = []  # list of PartialCorrelations

    def add_test(self, partial_corr):
        self.tests.append(partial_corr)


# Information of all partial correlations
class GraphCorrelations:
    def __init__(self, init_node_correlations):
        self.node_correlations = init_node_correlations  # array of CorrelationTest for each pair

    def get_correlation(self, i, j, dep_set):
        for p in self.node_correlations[i][j].tests:
            if set(p.on_set) == set(dep_set):
                # print(set(p.on_set), set(dep_set))
                return p.corr
        return None

    def find_best_h(self, i, j, dep_set):
        for h in dep_set:
            reduced_set = list(set(dep_set) - {h})  # list(set(dep_set.variables) - set(h))
            if self.get_correlation(i, j, reduced_set) is not None:
                return h
        return dep_set[0]

    def calculate_correlation(self, i, j, dep_set):
        corr = self.get_correlation(i, j, dep_set)
        if corr is not None:
            return corr
        else:
            h = self.find_best_h(i, j, dep_set)
            reduced_set = list(set(dep_set) - {h})  # list(set(dep_set.variables) - set(h))
            prev_corr = self.calculate_correlation(i, j, reduced_set)
            i_corr = self.calculate_correlation(i, h, reduced_set)
            j_corr = self.calculate_correlation(j, h, reduced_set)
            corr = (prev_corr - i_corr * j_corr) / \
                   (np.sqrt(1 - np.power(i_corr, 2)) * np.sqrt(1 - np.power(j_corr, 2)))
            (self.node_correlations[i][j]).add_test(PartialCorrelation(s=dep_set, corr=corr))
            return corr


def z_trans(rho):
    return 0.5 * np.log((1 + rho) / (1 - rho))


def all_neighbours(i, adj, nodes):
    neighbours = []
    for j in range(nodes):
        if adj[i, j] == 1:
            neighbours.append(j)
    return neighbours


def chose_list(neighbours, l_size):
    # Could Throw Exception in: l_size > len(neighbours)
    all_sets = list(itertools.combinations(neighbours, l_size))
    return all_sets


def adjacent_sets(i, j, l_i, adj, nodes):
    # Could Throw Exception in: j is adjacent with i
    if l_i == 0:
        return [()]
    neighbours = all_neighbours(i, adj, nodes)
    neighbours = list(set(neighbours) - {j})
    # all_sets.append( new list in size l_i in neighbours )
    # print("i", i, " j", j, " neigh:", neighbours)
    return chose_list(neighbours, l_i)


def all_pairs(adj, l_i, nodes):
    pairs = []
    for i in range(nodes):
        i_neighbour_size = len(all_neighbours(i, adj, nodes))
        if i_neighbour_size >= l_i + 1:
            for j in range(nodes):
                if adj[i, j] == 1:
                    pairs.append([i, j])
    return pairs


def show_graph(adj, nodes):
    for r in range(nodes):
        print adj[r][:]


n = 200  # Number of Nodes(Vertices)
file_name = 'pc-data.csv'
file_data = pd.read_csv(file_name, sep='\t', header=None)
v = (np.array(file_data.values)).T  # Read data of nodes
data_size = v[0].size  # = 5000
accept_level = 2.3263  # t_n > Q^-1(alpha/2) = Q^-1(0.01) = 2.3263
adjacency = np.ones(shape=(n, n))
for cnt in range(n):
    adjacency[cnt, cnt] = 0

# Initializing with zero order tests
init_correlations = [[CorrelationTests() for col in range(n)] for row in range(n)]
for row in range(n):
    for col in range(row, n):  # corr(xi, xi) is calculated and it's corrcoeff = 1
        test = PartialCorrelation([], np.corrcoef(v[row], v[col])[1, 0])
        init_correlations[row][col].add_test(test)
        if row != col:
            init_correlations[col][row].add_test(test)

g = GraphCorrelations(init_correlations)
separator = [[[] for col in range(n)] for row in range(n)]
normal_level = accept_level / np.sqrt(data_size - 3)

# Start
for l in range(n):
    # new pair (v1,v2), v1 and v2 are adjacent, adj(v1)\v2 >= l
    v_s = all_pairs(adjacency, l, n)
    for pair in tqdm(v_s):
        v1 = pair[0]
        v2 = pair[1]
        v1_adjacent = adjacent_sets(v1, v2, l, adjacency, n)
        for kk in v1_adjacent:
            k = list(kk)
            corr_coef = g.calculate_correlation(v1, v2, k)
            z_rho = z_trans(corr_coef)
            if np.abs(z_rho) < normal_level:
                # print "YoHoo! v1=", v1, " and v2=", v2, " are cut by k=", k
                separator[v1][v2] = k
                separator[v2][v1] = k
                adjacency[v1][v2] = 0
                adjacency[v2][v1] = 0
                break
        # if adjacency[v1, v2] == 0:
        #     break

print "Finally, DAG Edges Are:"
for row in range(n):
    for col in range(row + 1, n):
        if adjacency[row][col] == 1:
            print "(", row, ",", col, ")"
# np.savetxt("adjacency.txt", np.int32(adjacency))
