import numpy as np
import networkx as nx
import igraph as ig
from sklearn.metrics.cluster import pair_confusion_matrix
import time
from multiprocessing import Pool
import copy
import os
import random
import argparse


class Partition(object):
    def __init__(self, sample_perc=.5, init_partition=None):
        np.random.seed()
        self.n_nodes = G_ig.vcount()
        self.n_edges = G_ig.ecount()
        self.sample_size = int(self.n_nodes * sample_perc)
        self.membership = []
        self.n_comms = 0
        self.fitness = None
        if init_partition is None:
            self.initialize_partition()
        else:
            self.membership = init_partition
            self.n_comms = len(np.unique(self.membership))

    def initialize_partition(self):
        if random.uniform(0, 1) > .5:
            # sample nodes
            subsample = np.random.choice(self.n_nodes, size=self.sample_size, replace=False)
            subgraph = G_ig.subgraph(subsample)
        else:
            # sample edges
            subsample = np.random.choice(self.n_edges, size=self.sample_size, replace=False)
            subgraph = G_ig.subgraph_edges(subsample)
        subsample_partition_memb = np.zeros(self.n_nodes) - 1
        subsample_nodes = [v.index for v in subgraph.vs]
        # leiden on subgraph
        subsample_subpartition = subgraph.community_leiden(objective_function='modularity', weights=None, beta=0.01,
                                                           initial_membership=None, n_iterations=2)
        subsample_subpartition_memb = subsample_subpartition.membership
        subsample_partition_memb[subsample_nodes] = subsample_subpartition_memb
        first_available_comm_id = np.max(subsample_subpartition_memb) + 1
        arg_unassigned = subsample_partition_memb == -1
        subsample_partition_memb[arg_unassigned] = list(range(first_available_comm_id,
                                                              first_available_comm_id + sum(arg_unassigned)))
        self.membership = subsample_partition_memb.astype(int)
        self.n_comms = np.max(self.membership)+1

    def optimize(self, iters):
        # leiden
        partition = G_ig.community_leiden(objective_function='modularity', weights=None, beta=0.01,
                                       initial_membership=self.membership, n_iterations=iters)
        self.membership = partition.membership
        self.n_comms = np.max(self.membership) + 1
        self.fitness = partition.modularity
        return self

    def mutate(self):
        self.membership = np.array(self.membership)
        mut_type = np.random.choice([1, 2])
        if mut_type == 1:
            # split a community
            comm_id_to_split = np.random.choice(self.n_comms)
            rel_idx = np.where(self.membership == comm_id_to_split)[0]
            if len(rel_idx) < 2:
                pass
            else:
                # newman
                subgraph = G_ig.subgraph(rel_idx)
                new_assignment = subgraph.community_leading_eigenvector(clusters=2).membership
                new_assignment[new_assignment == 0] = comm_id_to_split
                new_assignment[new_assignment == 1] = self.n_comms
                self.membership[self.membership == comm_id_to_split] = new_assignment
                self.n_comms += 1
        else:
            # randomly unify two connected communities
            candidate_edges = np.random.choice(G_ig.es, size=10, replace=False)
            for e in candidate_edges:
                v1, v2 = e.tuple
                comm1, comm2 = self.membership[v1], self.membership[v2]
                if comm1 == comm2: continue
                self.membership[self.membership == comm1] = comm2
                self.membership[self.membership == self.n_comms-1] = comm1
                self.n_comms -= 1
                break
        return self


def load_graph(path):
    nx_graph = nx.read_adjlist(path)
    mapping = dict(zip(nx_graph, range(nx_graph.number_of_nodes())))
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
    return ig_graph, nx_graph


def compute_partition_similarity(partition_a, partition_b):
    conf = pair_confusion_matrix(partition_a, partition_b)
    b, d, c, a = conf.flatten()
    jac = a/(a+c+d)
    return jac


def overlap(partition_memberships):
    consensus_partition = partition_memberships[0]
    n_nodes = len(consensus_partition)
    for i, partition in enumerate(partition_memberships[1:]):
        partition = partition
        cluster_id_mapping = {}
        c = 0
        for node_id in range(n_nodes):
            cur_pair = (consensus_partition[node_id], partition[node_id])
            if cur_pair not in cluster_id_mapping:
                cluster_id_mapping[cur_pair] = c
                c += 1
            consensus_partition[node_id] = cluster_id_mapping[cur_pair]
    return consensus_partition


def create_population(size_of_population):
    sample_perc_per_indiv = np.random.uniform(.2, .9, size=size_of_population)
    params = [samp_perc for samp_perc in sample_perc_per_indiv]
    pool = Pool(min(len(params), N_WORKERS))
    results = [pool.apply_async(Partition, (samp_perc,)) for samp_perc in params]
    pool.close()
    pool.join()
    pop = [x.get() for x in results]
    return pop


def get_probabilities(values):
    """
    Calculates probabilities.
    :param values: list of ranks.
    :return: probability function (list).
    """
    p = []
    values = np.max(values) + 1 - values
    denom = np.sum(values ** SELECTION_POWER)
    for value in values:
        p.append(value ** SELECTION_POWER / denom)
    p = (p-np.min(p))/(np.max(p)-np.min(p))
    p = p/np.sum(p)
    return p


def single_crossover(indiv1, indiv2):
    single_offspring = Partition(init_partition=overlap([indiv1.membership, indiv2.membership]))
    return single_offspring


def pop_crossover_and_immigiration(population, n_offspring, n_immig):
    idx_to_cross = []
    as_is_offspring = []
    for i in range(n_offspring):
        idx1, idx2 = np.random.choice(len(population), size=2, replace=False, p=PROBS)
        if random.uniform(0, 1) > .5: idx_to_cross.append([idx1, idx2])
        else: as_is_offspring.append(population[idx1])

    immigs = create_population(size_of_population=n_immig)
    pool = Pool(N_WORKERS)
    results = [pool.apply_async(single_crossover, (population[idx1], population[idx2])) for idx1, idx2 in idx_to_cross]
    pool.close()
    pool.join()
    crossed_offspring = [x.get() for x in results]
    offspring = crossed_offspring + as_is_offspring
    return offspring, immigs


def run_ga():
    last_best_memb = []
    stopping_criterion = 10  # generations
    cnt_convergence = 0

    # Population initialization
    pop = create_population(size_of_population=POPULATION_SIZE)

    for gen in range(1, MAX_GENS+1):
        start_time = time.time()

        # Population optimization
        pool = Pool(N_WORKERS)
        results = [pool.apply_async(indiv.optimize, (3, )) for indiv in pop]
        pool.close()
        pool.join()
        pop = [x.get() for x in results]

        pop_fit = [indiv.fitness for indiv in pop]
        best_score = np.max(pop_fit)
        best_indiv = pop[np.argmax(pop_fit)]

        # stopping criteria related
        if last_best_memb:
            sim_to_last_best = compute_partition_similarity(best_indiv.membership, last_best_memb)
            if sim_to_last_best > .98: cnt_convergence += 1
            else: cnt_convergence = 0
        last_best_memb = copy.deepcopy(best_indiv.membership)
        if cnt_convergence == stopping_criterion: break
        pop_rank_by_fitness = np.argsort(pop_fit)[::-1]
        pop = [pop[i] for i in pop_rank_by_fitness]
        if gen == MAX_GENS: break

        # elitist selection
        elite_idx = [0]
        idx = 1
        threshold = .9
        while len(elite_idx) < N_ELITE:
            potential_elite = pop[idx]
            elite_flag = True
            for elite_indiv_idx in elite_idx:
                jac = compute_partition_similarity(potential_elite.membership, pop[elite_indiv_idx].membership)
                if jac > threshold:
                    elite_flag = False
                    break
            if elite_flag:
                elite_idx.append(idx)
            idx += 1
        elite = [pop[i] for i in elite_idx]
        pop = elite + [pop[i] for i in range(POPULATION_SIZE) if i not in elite_idx]

        # Population crossover and immigration
        offspring, immigs = pop_crossover_and_immigiration(pop, n_offspring=POPULATION_SIZE-N_ELITE-N_IMMIG,
                                                           n_immig=N_IMMIG)
        pool = Pool(min(len(offspring), N_WORKERS))
        results = [pool.apply_async(indiv.mutate, ()) for indiv in offspring]
        pool.close()
        pool.join()
        offspring = [x.get() for x in results]
        print(gen, 'Top fitness:', np.round(best_score, 6), '; Average fitness:',
              np.round(np.mean(pop_fit), 6), '; Time per generation:', np.round(time.time() - start_time, 2),
              '; convergence:', cnt_convergence)
        pop = elite + offspring + immigs

    # return best
    return pop[0]


G_ig = None
POPULATION_SIZE, N_WORKERS, MAX_GENS, N_IMMIG, N_ELITE, SELECTION_POWER = -1, -1, -1, -1, -1, 5
PROBS = []

if __name__ == "__main__":
    # parse script parameters
    parser = argparse.ArgumentParser(description='TAU')
    # general parameters
    parser.add_argument('--graph', type=str, help='path to graph file; supports adjacency list format')
    parser.add_argument('--size', type=int, default=60, help='size of population; default is 60')
    parser.add_argument('--workers', type=int, default=-1, help='number of workers; default is number of available CPUs')
    parser.add_argument('--max_generations', type=int, default=500, help='maximum number of generations to run;'
                                                                         ' default is 500')
    args = parser.parse_args()

    g, _ = load_graph(args.graph)
    print(args.graph, g.vcount(), g.ecount())
    population_size = args.size
    if population_size < 10: population_size = 10
    p_elite, p_immig = .1, .15

    # set global variable values
    cpus = os.cpu_count()
    N_WORKERS = min(cpus, population_size) if args.workers == -1 else np.min([cpus, population_size, args.workers])
    PROBS = get_probabilities(np.arange(population_size))
    N_ELITE, N_IMMIG = int(p_elite * population_size), int(p_immig * population_size)
    G_ig = g
    POPULATION_SIZE = population_size
    MAX_GENS = args.max_generations

    print('Parameter values: pop_size=%d, workers=%d, max_generations=%d' %(POPULATION_SIZE, N_WORKERS, MAX_GENS))

    best_partition = run_ga()
    np.save('%s_best_partition_membership.npy' % args.graph, best_partition.membership)
