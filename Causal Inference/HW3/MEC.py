import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm


# Function Adjacent:
def adjacent_orienting(g, a, b):
    adj = []
    new_g = g.copy()
    for bb in b:
        flag = False
        for aa in a:
            if g[aa][bb] == 1:  # or g[bb][aa] == 1 is not necessary
                flag = True
                new_g[bb][aa] = 0  # directing edge to aa-->bb
        if flag:
            adj.append(bb)
    # g_t = list(set(b) - set(adj))
    return new_g, adj


# Orient x-->y--z s
def meek_orient(g, a, yz_pair):
    new_g = g.copy()
    for aa in a:
        for yz in yz_pair:
            y = yz[0]
            z = yz[1]
            if new_g[aa][y] == 1:
                if new_g[aa][z] == 0:
                    new_g[z][y] = 0  # directing y-->z
            if new_g[aa][z] == 1:
                if new_g[aa][y] == 0:
                    new_g[y][z] = 0  # directing z-->y
    return new_g


# Check Same Neighbours
def is_neighbour(pair1, pair2):
    y1 = pair1[0]
    z1 = pair1[1]
    y2 = pair2[0]
    z2 = pair2[1]
    if y1 == y2 or y1 == z2 or z1 == y2 or z1 == z2:
        return True
    return False


# Find an index for 1 in Matrix
def place_of_one(matrix):
    l_matrix = matrix.shape[0]
    for ii in range(l_matrix):
        for jj in range(ii + 1, l_matrix):
            if matrix[ii][jj]:
                return True, ii, jj
    return False, 0, 0


# Mix Undirected Nodes if Neighbour, to Get new UCCGs in pair
def mix_neighbours(pairs):
    l_pairs = len(pairs)
    pre_adjacent = np.zeros(shape=(l_pairs, l_pairs), dtype=bool)
    for ii in range(l_pairs):
        for jj in range(ii + 1, l_pairs):
            if is_neighbour(pairs[ii], pairs[jj]):
                pre_adjacent[ii][jj] = True
                pre_adjacent[jj][ii] = True

    new_pairs = pairs
    flag, ii, jj = place_of_one(pre_adjacent)
    while flag:
        pre_adjacent[ii][:] = np.logical_or(pre_adjacent[ii][:], pre_adjacent[jj][:])
        pre_adjacent[:][ii] = np.logical_or(pre_adjacent[:][ii], pre_adjacent[:][jj])
        pre_adjacent[ii][ii] = False
        pair1 = pairs[ii]
        pair2 = pairs[jj]
        pairs[ii] = list(pair1) + list(set(pair2) - set(pair1))
        new_pairs.remove(pair2)
        pre_adjacent = np.delete(pre_adjacent, jj, axis=0)
        pre_adjacent = np.delete(pre_adjacent, jj, axis=1)
        flag, ii, jj = place_of_one(pre_adjacent)
    return new_pairs


# List of Nodes to UCCGs
def nodes2uccg(nodes, g):
    l_nodes = len(nodes)
    graph = np.zeros(shape=(l_nodes, l_nodes))
    for ii in range(l_nodes):
        for jj in range(ii + 1, l_nodes):
            if g[nodes[ii]][nodes[jj]] == 1:  # No Difference than g[jj][ii]
                graph[ii][jj] = 1
                graph[jj][ii] = 1
    return graph


# Find Undirected Arrows
def undirected(g, t_pairs):
    undirected_pairs = []
    for pair in t_pairs:
        y = pair[0]
        z = pair[1]
        if g[y][z] == 1 and g[z][y] == 1:
            undirected_pairs.append(pair)

    mixed_nodes = mix_neighbours(undirected_pairs)
    graphs = []
    for nodes in mixed_nodes:
        graphs.append(nodes2uccg(nodes, g))
    return graphs


# Function ChainCom(U, v):
def chain_com(u, v, p):
    a = [v]
    b = list(set(range(p)) - set(a))
    g = u.copy()
    o = []
    flag = True
    while flag:  # if B is not empty
        g, t = adjacent_orienting(g, a, b)
        yz_pair = list(itertools.combinations(t, 2))
        g = meek_orient(g, a, yz_pair)
        a = t  # A = T
        flag = (set(b) != set(t))
        b = list(set(b) - set(t))
        o = o + undirected(g, yz_pair)
    return o


# Function SizeMEC(U):
def size_mec(u):
    # u is an undirected and connected chordal graph (UCCGs)
    # u is the adjacency matrix, and symmetric, p * p
    p = u.shape[0]  # p = u.shape[0] = u.shape[1]
    n = np.sum(u) / 2  # number of existing edges
    if n == p - 1:
        return p
    if n == p:
        return 2 * p
    max_n = p * (p - 1) / 2
    if n == max_n - 2:
        return (p * p - p - 4) * math.factorial(p - 3)
    if n == max_n - 1:
        return 2 * math.factorial(p - 1) - math.factorial(p - 2)
    if n == max_n:
        return math.factorial(p)
    s = np.zeros(shape=(p, 1), dtype=int)
    for j in range(p):
        sub_rooted_chains = chain_com(u, j, p)
        for sub_u in sub_rooted_chains:
            sub_mec_size = size_mec(sub_u)
            s[j] = s[j] + sub_mec_size
    return np.sum(s)


# MECs to UCCSs
def mec2uccgs(mec):
    l_mec = mec.shape[0]
    all_nodes = range(l_mec)
    all_pairs = list(itertools.combinations(all_nodes, 2))
    graphs = undirected(mec, all_pairs)
    return graphs


def size_cpdag(pdag):
    related_uccgs = mec2uccgs(pdag)
    whole_size = 1
    for uccg in related_uccgs:
        whole_size = whole_size * size_mec(uccg)
    return whole_size


def pdag_initializer(p):
    # Returns an open forward chained DAG
    pdag = np.zeros(shape=(p, p))
    for ii in range(p - 1):
        pdag[ii][ii + 1] = 1
    n_e = p - 1
    return pdag, n_e


def various_pairs(dag, p):
    non_adjacents = []
    undirected_pairs = []
    directed_pairs = []
    v_structures = []
    possible_v = []
    for ii in range(p):
        for jj in range(ii + 1, p):
            kk_range = range(p)
            kk_range.remove(ii)
            kk_range.remove(jj)
            if dag[ii][jj] == 1:
                if dag[jj][ii] == 1:
                    undirected_pairs.append([ii, jj])
                    for kk in kk_range:
                        if dag[jj][kk] == 1 and dag[kk][jj] == 1 and dag[ii][kk] == 0 and dag[kk][ii] == 0:
                            possible_v.append([ii, kk, jj])
                        if dag[ii][kk] == 1 and dag[kk][ii] == 1 and dag[jj][kk] == 0 and dag[kk][jj] == 0:
                            possible_v.append([jj, kk, ii])
                else:
                    directed_pairs.append([ii, jj])
                    for kk in kk_range:
                        if dag[kk][jj] == 1 and dag[jj][kk] == 0 and dag[ii][kk] == 0 and dag[kk][ii] == 0:
                            v_structures.append([ii, kk, jj])
            else:
                if dag[jj][ii] == 1:
                    directed_pairs.append([jj, ii])
                    for kk in kk_range:
                        if dag[kk][jj] == 0 and dag[jj][kk] == 0 and dag[ii][kk] == 0 and dag[kk][ii] == 1:
                            v_structures.append([jj, kk, ii])
                else:
                    non_adjacents.append([ii, jj])
    return non_adjacents, undirected_pairs, directed_pairs,\
        possible_v, v_structures


def is_clique(pdag, nodes):
    edges = list(itertools.combinations(nodes, 2))
    for pair in edges:
        x = pair[0]
        y = pair[1]
        if pdag[x][y] == 0 or pdag[y][x] == 0:
            return False
    return True


def neighbours(pdag, p, node):
    neighs = []
    for ii in range(p):
        if pdag[ii][node] == 1 and pdag[node][ii] == 1:
            neighs.append(ii)
    return neighs


def parents(pdag, p, node):
    pars = []
    for ii in range(p):
        if pdag[ii][node] == 1 and pdag[node][ii] == 0:
            pars.append(ii)
    return pars


def does_path_cross(set_in, set_out, set_block, pdag, is_undirected=True):
    return True


def is_insert_u_valid(x, y, pdag, p):
    par_x = parents(pdag, p, node=x)
    par_y = parents(pdag, p, node=y)
    is_same_parents = (par_x == par_y)
    n_x = neighbours(pdag, p, x)
    n_y = neighbours(pdag, p, y)
    n_xy = list(set(n_x).intersection(n_y))
    n_x_pure = list(set(n_x) - set(n_xy))
    n_y_pure = list(set(n_y) - set(n_xy))
    # if undirected paths from n_x_pure to n_y_pure contains n_xy:
    is_paths_contained = does_path_cross(set_in=n_x_pure, set_out=n_y_pure,
                                         set_block=n_xy, pdag=pdag, is_undirected=True)
    return is_same_parents and is_paths_contained


def is_insert_d_valid(x, y, pdag, p):
    par_x = parents(pdag, p, node=x)
    par_y = parents(pdag, p, node=y)
    is_not_same_parents = (par_x != par_y)
    n_y = neighbours(pdag, p, y)
    omega_xy = list(set(par_x).intersection(n_y))
    # n_x_pure = list(set(n_x) - set(n_xy))
    n_y_pure = list(set(n_y) - set(omega_xy))
    # if partially directed paths from n_y_pure to x contains omega_xy
    is_paths_contained = does_path_cross(set_in=n_y_pure, set_out=x,
                                         set_block=omega_xy, pdag=pdag, is_undirected=False)
    return is_not_same_parents and is_paths_contained and is_clique(pdag, omega_xy)


def is_rempve_v_valid(x, y, z, pdag, p):
    par_x = parents(pdag, p, node=x)
    par_y = parents(pdag, p, node=y)
    is_same_parents = (par_x == par_y)
    n_x = neighbours(pdag, p, x)
    n_y = neighbours(pdag, p, y)
    n_xy = list(set(n_x).intersection(n_y))
    set_1 = list(par_x) + list(n_xy)
    list_xy = [x, y]
    set_2 = list(set(parents(pdag, p, z)) - set(list_xy))
    is_rule2_valid = (set_1 == set_2)
    is_crossed = does_path_cross(set_in=x, set_out=y, set_block=n_xy, pdag=pdag, is_undirected=True)
    return is_same_parents and is_rule2_valid and is_crossed


def set_of_valid_operations(dag, p, n, n_max):
    o = []
    non_adj_pairs, undirected_pairs, directed_pairs, undirected_v_structures, v_structures = various_pairs(dag, p)
    # DeleteU
    for pair in undirected_pairs:
        x = pair[0]
        y = pair[1]
        n_x = neighbours(dag, p, x)
        n_y = neighbours(dag, p, y)
        n_xy = list(set(n_x).intersection(n_y))
        if is_clique(dag, n_xy):  # Nxy must be clique
            o.append([1, x, y, 0])
    # DeleteD
    for pair in directed_pairs:
        x = pair[0]
        y = pair[1]
        n_y = neighbours(dag, p, y)
        if is_clique(dag, n_y):
            o.append([3, x, y, 0])
    # RemoveV
    for v_structure in v_structures:
        x = v_structure[0]
        y = v_structure[1]
        z = v_structure[2]
        if is_rempve_v_valid(x, y, z, dag, p):
            o.append([5, x, y, z])
    # MakeV
    for v_structure in undirected_v_structures:
        x = v_structure[0]
        y = v_structure[1]
        z = v_structure[2]
        n_x = neighbours(dag, p, x)
        n_y = neighbours(dag, p, y)
        n_xy = list(set(n_x).intersection(n_y))
        if does_path_cross(set_in=x, set_out=y, set_block=n_xy, pdag=dag, is_undirected=True):
            o.append([4, x, y, z])
    # Inserts
    if n < n_max:
        for pair in non_adj_pairs:
            x = pair[0]
            y = pair[1]
            # InsertU
            if is_insert_u_valid(x, y, pdag=dag, p=p):
                o.append([0, x, y, 0])
            # InsertD, x->y
            if is_insert_d_valid(x, y, pdag=dag, p=p):
                o.append([2, x, y, 0])
            # InsertD, y->x
            if is_insert_d_valid(y, x, pdag=dag, p=p):
                o.append([2, y, x, 0])
    return o


def to_next_state(state, op):
    kind = op[0]  # {0: insertU, 1: deleteU, 2: insetD, 3: deleteD, 4: makeV, 5: removeV}
    x = op[1]
    y = op[2]
    z = op[3]
    next_state = state.copy()
    if kind == 0:  # insertU
        next_state[x][y] = 1
        next_state[y][x] = 1
    elif kind == 1:  # deleteU
        next_state[x][y] = 0
        next_state[y][x] = 0
    elif kind == 2:  # insertD
        next_state[x][y] = 1
    elif kind == 3:  # deleteD
        next_state[x][y] = 0
    elif kind == 4:  # makeV
        next_state[x][z] = 1
        next_state[y][z] = 1
    else:  # removeV
        next_state[x][z] = 0
        next_state[y][z] = 0
    return next_state


def markov_chain(p, n_itter=10000):
    max_n = p * (p - 1) / 2
    max_mec_size = 1000  # Not Sure
    e, n_e = pdag_initializer(p)
    sum_mt = 0
    probability_surface = np.zeros(shape=(max_n, max_mec_size))
    for ii in tqdm(range(n_itter)):
        # Size of CPDAG
        size_e = size_cpdag(e)
        # Set of Operations, each one is like: [kind, node1, node2, none/node3]
        o = set_of_valid_operations(e, p, n_e, max_n)
        # Number of possible operations
        mt = len(o)
        probability_surface[n_e][size_e] = probability_surface[n_e][size_e] + 1 / np.float32(mt)
        sum_mt = sum_mt + 1 / mt
        # Chose an operation
        o_selected_index = np.random.random_integers(0, mt - 1)
        o_selected = o[o_selected_index][:]
        # Apply the operand
        kind = o_selected[0]
        if kind == 0 or kind == 2:  # insertU or insertD
            n_e = n_e + 1
        elif kind == 1 or kind == 3:  # deleteU or deleteD
            n_e = n_e - 1
        e = to_next_state(e, o_selected)
    fig = plt.figure()
    ax = Axes3D(fig)
    # fig.add_subplot(111, projection='3d')
    x_range = range(max_n)
    y_range = range(max_mec_size)
    y_plot, x_plot = np.meshgrid(y_range, x_range)
    ax.plot_wireframe(X=y_plot, Y=x_plot, Z=probability_surface)
    # Axes3D.plot_surface(ax, X=x_plot, Y=y_plot, Z=probability_surface)
    # Axes3D.contour(X=x_plot, Y=y_plot, Z=probability_surface)
    plt.show()
    return 1


# Main Test
# Part 1
mec_in = np.array([[0, 1, 0, 0, 0],
                   [1, 0, 1, 1, 1],
                   [0, 1, 0, 0, 1],
                   [0, 1, 0, 0, 1],
                   [0, 1, 1, 1, 0]])

# related_uccgs = mec2uccgs(mec_in)
# total_size = 1
# for uccg in related_uccgs:
#     total_size = total_size * size_mec(uccg)

# --> use size_mec(mec) if mec is an uccg
total_size = size_cpdag(mec_in)
print total_size

# Part 2
markov_chain(10)
