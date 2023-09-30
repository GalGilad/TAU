import numpy as np
import networkx as nx
import igraph as ig
from sklearn.metrics.cluster import pair_confusion_matrix
import time
from multiprocessing import Pool
import itertools
import os
import random
import argparse


class Partition:
    def __init__(self, sample_fraction=.5, init_partition=None):
        np.random.seed()
        self.n_nodes = G_ig.vcount()
        self.n_edges = G_ig.ecount()
        self.sample_size_nodes = int(self.n_nodes * sample_fraction)
        self.sample_size_edges = int(self.n_edges * sample_fraction)
        self.membership = []
        self.n_comms = 0
        self.fitness = None
        if init_partition is None:
            self.initialize_partition()
        else:
            self.membership = init_partition
            self.n_comms = len(np.unique(self.membership))

    def initialize_partition(self):
        if flip_coin():
            # sample nodes
            subsample = np.random.choice(self.n_nodes, size=self.sample_size_nodes, replace=False)
            subgraph = G_ig.subgraph(subsample)
        else:
            # sample edges
            subsample = np.random.choice(self.n_edges, size=self.sample_size_edges, replace=False)
            subgraph = G_ig.subgraph_edges(subsample)
        subsample_partition_memb = np.zeros(self.n_nodes) - 1
        subsample_nodes = [v.index for v in subgraph.vs]
        # leiden on subgraph
        subsample_subpartition = subgraph.community_leiden(objective_function='modularity')
        subsample_subpartition_memb = subsample_subpartition.membership
        subsample_partition_memb[subsample_nodes] = subsample_subpartition_memb
        first_available_comm_id = np.max(subsample_subpartition_memb) + 1
        arg_unassigned = subsample_partition_memb == -1
        subsample_partition_memb[arg_unassigned] = list(range(first_available_comm_id,
                                                              first_available_comm_id + sum(arg_unassigned)))
        self.membership = subsample_partition_memb.astype(int)
        self.n_comms = np.max(self.membership)+1

    def optimize(self):
        # leiden
        partition = G_ig.community_leiden(objective_function='modularity', initial_membership=self.membership,
                                          n_iterations=3)
        self.membership = partition.membership
        self.n_comms = np.max(self.membership) + 1
        self.fitness = partition.modularity
        return self

    def newman_split(self, indices, comm_id_to_split):
        # newman
        subgraph = G_ig.subgraph(indices)
        new_assignment = subgraph.community_leading_eigenvector(clusters=2).membership
        new_assignment[new_assignment == 0] = comm_id_to_split
        new_assignment[new_assignment == 1] = self.n_comms
        self.membership[self.membership == comm_id_to_split] = new_assignment

    def random_split(self, indices):
        size_to_split = min(1, np.random.choice(len(indices)//2))
        idx_to_split = np.random.choice(indices, size=size_to_split, replace=False)
        self.membership[idx_to_split] = self.n_comms

    def mutate(self):
        self.membership = np.array(self.membership)
        if flip_coin():
            # split a community
            comm_id_to_split = np.random.choice(self.n_comms)
            idx_to_split = np.where(self.membership == comm_id_to_split)[0]
            if len(idx_to_split) > 2:
                min_comm_size_newman = 10
                if len(idx_to_split) > min_comm_size_newman:
                    if flip_coin():
                        self.newman_split(idx_to_split, comm_id_to_split)
                    else:
                        self.random_split(idx_to_split)
                else:
                    self.random_split(idx_to_split)
                self.n_comms += 1
        else:
            # randomly merge two connected communities
            candidate_edges = random.choices(G_ig.es, k=10)
            for i, e in enumerate(candidate_edges):
                v1, v2 = e.tuple
                comm1, comm2 = self.membership[v1], self.membership[v2]
                if comm1 == comm2:
                    continue
                self.membership[self.membership == comm1] = comm2
                self.membership[self.membership == self.n_comms-1] = comm1
                self.n_comms -= 1
                break
        return self


def flip_coin():
    return random.uniform(0, 1) > .5


def load_graph(path):
    nx_graph = nx.read_adjlist(path)
    mapping = dict(zip(nx_graph.nodes(), range(nx_graph.number_of_nodes())))
    nx_graph = nx.relabel_nodes(nx_graph, mapping)
    ig_graph = ig.Graph(len(nx_graph), list(zip(*list(zip(*nx.to_edgelist(nx_graph)))[:2])))
    return ig_graph


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
    sample_fraction_per_indiv = np.random.uniform(.2, .9, size=size_of_population)
    params = [sample_fraction for sample_fraction in sample_fraction_per_indiv]
    pool = Pool(min(size_of_population, N_WORKERS))
    results = [pool.apply_async(Partition, (sample_fraction,)) for sample_fraction in params]
    pool.close()
    pool.join()
    population = [x.get() for x in results]
    return population


def get_probabilities(values):
    p = []
    values = np.max(values) + 1 - values
    denom = np.sum(values ** SELECTION_POWER)
    for value in values:
        p.append(value ** SELECTION_POWER / denom)
    return p


def single_crossover(idx1, idx2):
    partitions_overlap = overlap([pop[idx1].membership, pop[idx2].membership])
    single_offspring = Partition(init_partition=partitions_overlap)
    return single_offspring


def pop_crossover_and_immigration(n_offspring):
    idx_to_cross = []
    as_is_offspring = []
    for i in range(n_offspring):
        idx1, idx2 = np.random.choice(len(pop), size=2, replace=False, p=PROBS)
        if flip_coin():
            idx_to_cross.append([idx1, idx2])
        else:
            as_is_offspring.append(pop[idx1])
    pool = Pool(N_WORKERS)
    results = [pool.apply_async(single_crossover, (idx1, idx2)) for idx1, idx2 in idx_to_cross]
    pool.close()
    pool.join()
    crossed_offspring = [x.get() for x in results]
    offspring = crossed_offspring + as_is_offspring

    immigrants = create_population(size_of_population=N_IMMIGRANTS)

    return offspring, immigrants


def compute_partition_similarity_by_pop_idx(idx1, idx2):
    conf = pair_confusion_matrix(pop[idx1].membership, pop[idx2].membership)
    b, d, c, a = conf.flatten()
    jac = a/(a+c+d)
    return jac


def selection_helper_compute_similarities(combinations):
    assert 0 < len(combinations) <= N_WORKERS
    pool = Pool(len(combinations))
    results = [pool.apply_async(compute_partition_similarity_by_pop_idx, (idx1, idx2))
               for idx1, idx2 in combinations]
    pool.close()
    pool.join()
    similarities = [x.get() for x in results]
    similarities_dict = {tuple(sorted((idx1, idx2))): similarities[i] for i, (idx1, idx2) in enumerate(combinations)}
    return similarities_dict


def selection_helper_get_batch_of_pairs(elite_indices, candidate_idx, batch_size, computed=None):
    pairs = list(itertools.product(elite_indices, [candidate_idx]))
    pairs = [pair for pair in pairs if pair not in computed] if computed is not None else pairs
    batch_overflow = False if len(pairs) < batch_size else True
    i = 1
    while not batch_overflow:
        pairs += list(itertools.product(elite_indices, [candidate_idx+i]))
        for j in range(i):
            pairs.append((candidate_idx+j, candidate_idx+i))
        batch_overflow = False if len(pairs) < batch_size else True
        i += 1
    return pairs[:batch_size]


def elitist_selection(similarity_threshold):
    elite_indices, candidate_idx = [0], 1
    pairs_to_compute = selection_helper_get_batch_of_pairs(elite_indices=elite_indices, candidate_idx=candidate_idx,
                                                           batch_size=N_WORKERS, computed=[])
    similarities_between_solutions = selection_helper_compute_similarities(pairs_to_compute)
    computation_cycle_i, max_cycles = 1, 2
    computed_pairs = []
    while len(elite_indices) < N_ELITE and candidate_idx < len(pop):
        if computation_cycle_i == max_cycles:
            n_remaining = N_ELITE - len(elite_indices)
            elite_indices += list(np.random.choice(np.arange(candidate_idx, len(pop)), size=n_remaining, replace=False))
            break
        elite_flag = True
        for elite_idx in elite_indices:
            if (elite_idx, candidate_idx) not in similarities_between_solutions and computation_cycle_i < max_cycles:
                computation_cycle_i += 1
                computed_for_candidate = [(i, j) for (i, j) in computed_pairs
                                          if i >= candidate_idx or j >= candidate_idx]
                new_pairs = selection_helper_get_batch_of_pairs(elite_indices=elite_indices, candidate_idx=candidate_idx,
                                                                batch_size=N_WORKERS, computed=computed_for_candidate)
                similarities_between_solutions.update(selection_helper_compute_similarities(new_pairs))
            jac = similarities_between_solutions[elite_idx, candidate_idx]
            if jac > similarity_threshold:
                elite_flag = False
                break
        if elite_flag:
            elite_indices.append(candidate_idx)
        candidate_idx += 1
    return elite_indices


def find_partition():
    global pop
    last_best_memb = []
    best_modularity_per_generation = []
    cnt_convergence = 0

    # Population initialization
    pop = create_population(size_of_population=POPULATION_SIZE)

    for generation_i in range(1, MAX_GENERATIONS+1):
        start_time = time.time()

        # Population optimization
        pool = Pool(N_WORKERS)
        results = [pool.apply_async(indiv.optimize, ()) for indiv in pop]
        pool.close()
        pool.join()
        pop = [x.get() for x in results]

        pop_fit = [indiv.fitness for indiv in pop]
        best_score = np.max(pop_fit)
        best_modularity_per_generation.append(best_score)
        best_indiv = pop[np.argmax(pop_fit)]

        # stopping criteria related
        if last_best_memb:
            sim_to_last_best = compute_partition_similarity(best_indiv.membership, last_best_memb)
            if sim_to_last_best > stopping_criterion_jaccard:
                cnt_convergence += 1
            else:
                cnt_convergence = 0
        last_best_memb = best_indiv.membership
        if cnt_convergence == stopping_criterion_generations:
            break
        pop_rank_by_fitness = np.argsort(pop_fit)[::-1]
        pop = [pop[i] for i in pop_rank_by_fitness]
        if generation_i == MAX_GENERATIONS:
            break

        # elitist selection
        elite_idx = elitist_selection(elite_similarity_threshold)
        elite = [pop[i] for i in elite_idx]

        # crossover, immigration
        offspring, immigrants = pop_crossover_and_immigration(n_offspring=POPULATION_SIZE-N_ELITE-N_IMMIGRANTS)

        # mutation
        pool = Pool(min(len(offspring), N_WORKERS))
        results = [pool.apply_async(indiv.mutate, ()) for indiv in offspring]
        pool.close()
        pool.join()
        offspring = [x.get() for x in results]

        print(f'Generation {generation_i} Top fitness: {np.round(best_score, 6)}; Average fitness: '
              f'{np.round(np.mean(pop_fit), 6)}; Time per generation: {np.round(time.time() - start_time, 2)}; '
              f'convergence: {cnt_convergence}')
        pop = elite + offspring + immigrants

    # return best and modularity history
    return pop[0], best_modularity_per_generation


# globals and hyper-parameters
G_ig = None
POPULATION_SIZE = 60
N_WORKERS = 60
MAX_GENERATIONS = 500
N_IMMIGRANTS = -1
N_ELITE = -1
SELECTION_POWER = 5
PROBS = []
pop = []
p_elite = .1
p_immigrants = .15
stopping_criterion_generations = 10
stopping_criterion_jaccard = .98
elite_similarity_threshold = .9

if __name__ == "__main__":
    # parse script parameters
    parser = argparse.ArgumentParser(description='TAU')
    # general parameters
    parser.add_argument('--graph', type=str, help='path to graph file; supports adjacency list format')
    parser.add_argument('--size', type=int, default=60, help='size of population; default is 60')
    parser.add_argument('--workers', type=int, default=-1, help='number of workers; '
                                                                'default is number of available CPUs')
    parser.add_argument('--max_generations', type=int, default=500, help='maximum number of generations to run;'
                                                                         ' default is 500')
    args = parser.parse_args()

    # set global variable values
    g = load_graph(args.graph)
    population_size = max(10, args.size)
    cpus = os.cpu_count()
    N_WORKERS = min(cpus, population_size) if args.workers == -1 else np.min([cpus, population_size, args.workers])
    PROBS = get_probabilities(np.arange(population_size))
    N_ELITE, N_IMMIGRANTS = int(p_elite * population_size), int(p_immigrants * population_size)
    G_ig = g
    POPULATION_SIZE = population_size
    MAX_GENERATIONS = args.max_generations

    print(f'Main parameter values: pop_size={POPULATION_SIZE}, workers={N_WORKERS}, max_generations={MAX_GENERATIONS}')

    best_partition, mod_history = find_partition()
    np.save(f'TAU_partition_{args.graph}.npy', best_partition.membership)
