import numpy as np
import PCfuncs
from tqdm import tqdm


def random_dag(n, prob, lb, ub):
    ances = np.zeros(shape=(n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            p_rand = np.random.uniform()
            if p_rand < prob:
                weigth = lb + (ub - lb) * np.random.uniform()
                ances[i][j] = weigth
    return ances


def make_data(ances, noise, n):
    data = noise
    for i in range(1, n):
        for j in range(i):
            data[:, i] = data[:, i] + ances[j][i] * data[:, j]
    return data


def calc_diff(adj1, adj2, n):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(n):
        for j in range(n):
            if adj1[i][j]:
                if adj2[i][j]:
                    tp = tp + 1
                else:
                    tn = tn + 1
            else:
                if adj2[i][j]:
                    fp = fp + 1
                else:
                    fn = fn + 1
    return [tp, tn, fp, fn]


nodes = 20  # Number of Nodes
lB = 0.1
uB = 1
p = 0.2
iter_num = 200
normal_mean = np.zeros(shape=nodes)
normal_cov = np.identity(nodes)
samples = 1000
accept_levels = [2.807, 2.5758, 2.3263, 2.0537]  # 0.25%, 0.5% , 1%, 2%

# 0.25%
accept_level = accept_levels[0]
simple_pc_result = np.array([0, 0, 0, 0])
stable_pc_result = np.array([0, 0, 0, 0])
for cnt in tqdm(range(iter_num)):
    dag = random_dag(nodes, prob=p, lb=lB, ub=uB)
    exo_noise = np.random.multivariate_normal(mean=normal_mean, cov=normal_cov, size=samples)
    x_data = make_data(dag, exo_noise, nodes)
    real_adj = (dag > 0)
    simple_pc_adj = PCfuncs.pc(n=nodes, data=x_data, accept_level=accept_level)
    stable_pc_adj = PCfuncs.pc_stable(n=nodes, data=x_data, accept_level=accept_level)
    eval_simple_pc = calc_diff(real_adj, simple_pc_adj, nodes)
    eval_stable_pc = calc_diff(real_adj, stable_pc_adj, nodes)
    simple_pc_result = simple_pc_result + np.array(eval_simple_pc)
    stable_pc_result = stable_pc_result + np.array(eval_stable_pc)

simple_pc_result = 100 * np.array(simple_pc_result, dtype=float) / (iter_num * 400)
stable_pc_result = 100 * np.array(stable_pc_result, dtype=float) / (iter_num * 400)
print "For Accept Level = 0.25% :"
print "Simple PC [TP, TN, FP, FN]=", simple_pc_result, "Percent"
print "Stable PC [TP, TN, FP, FN]=", stable_pc_result, "Percent"

# 0.5%
accept_level = accept_levels[1]
simple_pc_result = np.array([0, 0, 0, 0])
stable_pc_result = np.array([0, 0, 0, 0])
for cnt in tqdm(range(iter_num)):
    dag = random_dag(nodes, prob=p, lb=lB, ub=uB)
    exo_noise = np.random.multivariate_normal(mean=normal_mean, cov=normal_cov, size=samples)
    x_data = make_data(dag, exo_noise, nodes)
    real_adj = (dag > 0)
    simple_pc_adj = PCfuncs.pc(n=nodes, data=x_data, accept_level=accept_level)
    stable_pc_adj = PCfuncs.pc_stable(n=nodes, data=x_data, accept_level=accept_level)
    eval_simple_pc = calc_diff(real_adj, simple_pc_adj, nodes)
    eval_stable_pc = calc_diff(real_adj, stable_pc_adj, nodes)
    simple_pc_result = simple_pc_result + np.array(eval_simple_pc)
    stable_pc_result = stable_pc_result + np.array(eval_stable_pc)

simple_pc_result = 100 * np.array(simple_pc_result, dtype=float) / (iter_num * 400)
stable_pc_result = 100 * np.array(stable_pc_result, dtype=float) / (iter_num * 400)
print "For Accept Level = 0.5% :"
print "Simple PC [TP, TN, FP, FN]=", simple_pc_result, "Percent"
print "Stable PC [TP, TN, FP, FN]=", stable_pc_result, "Percent"

# 1%
accept_level = accept_levels[2]
simple_pc_result = np.array([0, 0, 0, 0])
stable_pc_result = np.array([0, 0, 0, 0])
for cnt in tqdm(range(iter_num)):
    dag = random_dag(nodes, prob=p, lb=lB, ub=uB)
    exo_noise = np.random.multivariate_normal(mean=normal_mean, cov=normal_cov, size=samples)
    x_data = make_data(dag, exo_noise, nodes)
    real_adj = (dag > 0)
    simple_pc_adj = PCfuncs.pc(n=nodes, data=x_data, accept_level=accept_level)
    stable_pc_adj = PCfuncs.pc_stable(n=nodes, data=x_data, accept_level=accept_level)
    eval_simple_pc = calc_diff(real_adj, simple_pc_adj, nodes)
    eval_stable_pc = calc_diff(real_adj, stable_pc_adj, nodes)
    simple_pc_result = simple_pc_result + np.array(eval_simple_pc)
    stable_pc_result = stable_pc_result + np.array(eval_stable_pc)

simple_pc_result = 100 * np.array(simple_pc_result, dtype=float) / (iter_num * 400)
stable_pc_result = 100 * np.array(stable_pc_result, dtype=float) / (iter_num * 400)
print "For Accept Level = 1% :"
print "Simple PC [TP, TN, FP, FN]=", simple_pc_result, "Percent"
print "Stable PC [TP, TN, FP, FN]=", stable_pc_result, "Percent"

# 2%
accept_level = accept_levels[3]
simple_pc_result = np.array([0, 0, 0, 0])
stable_pc_result = np.array([0, 0, 0, 0])
for cnt in tqdm(range(iter_num)):
    dag = random_dag(nodes, prob=p, lb=lB, ub=uB)
    exo_noise = np.random.multivariate_normal(mean=normal_mean, cov=normal_cov, size=samples)
    x_data = make_data(dag, exo_noise, nodes)
    real_adj = (dag > 0)
    simple_pc_adj = PCfuncs.pc(n=nodes, data=x_data, accept_level=accept_level)
    stable_pc_adj = PCfuncs.pc_stable(n=nodes, data=x_data, accept_level=accept_level)
    eval_simple_pc = calc_diff(real_adj, simple_pc_adj, nodes)
    eval_stable_pc = calc_diff(real_adj, stable_pc_adj, nodes)
    simple_pc_result = simple_pc_result + np.array(eval_simple_pc)
    stable_pc_result = stable_pc_result + np.array(eval_stable_pc)

simple_pc_result = 100 * np.array(simple_pc_result, dtype=float) / (iter_num * 400)
stable_pc_result = 100 * np.array(stable_pc_result, dtype=float) / (iter_num * 400)
print "For Accept Level = 2% :"
print "Simple PC [TP, TN, FP, FN]=", simple_pc_result, "Percent"
print "Stable PC [TP, TN, FP, FN]=", stable_pc_result, "Percent"
