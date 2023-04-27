import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt('/Users/ssotomayorba/Documents/Personal/projects/machine-learning-projects/netflix/toy_data.txt')

'''
min_costs_k = np.array([])
ss = np.array([])
for k in range(1, 5):
    min_cost_k = np.inf
    min_s = np.nan
    for s in range(0, 5):
        mixture, post = common.init(X, k, s)
        mixture, post, cost = kmeans.run(X, mixture, post)
        if cost < min_cost_k:
            min_cost_k = cost
            min_s = s
    min_costs_k = np.append(min_costs_k, min_cost_k)
    ss = np.append(ss, min_s)

print(min_costs_k)
print(ss)

min_costs_k = np.array([])
ss = np.array([])
for k in range(1, 5):
    min_cost_k = np.inf
    min_s = np.nan
    for s in range(0, 5):
        mixture, post = common.init(X, k, s)
        mixture, post, cost = naive_em.run(X, mixture, post)
        if cost < min_cost_k:
            min_cost_k = cost
            min_s = s
    min_costs_k = np.append(min_costs_k, min_cost_k)
    ss = np.append(ss, min_s)

print(min_costs_k)
print(ss)

mixture, post = common.init(X, 4, 4)
mixture, post, cost = kmeans.run(X, mixture, post)
common.plot(X, mixture, post, 'Hola')

mixture, post = common.init(X, 4, 1)
mixture, post, cost = naive_em.run(X, mixture, post)
common.plot(X, mixture, post, 'Hola')
'''

best_bic = -np.inf
best_k = 0
s = [0, 1, 1, 1]
for k  in range(1, 5):
    mixture, post = common.init(X, k, s[k-1])
    mixture, post, cost = naive_em.run(X, mixture, post)
    bic = common.bic(X, mixture, cost)
    if bic > best_bic:
        best_bic = bic
        best_k = k
print(best_k, best_bic)