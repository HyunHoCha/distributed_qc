import numpy as np
import random
from gurobipy import *
import kahypar as kahypar

def is_unary_between(q1, q2, t1, t2, unary_list):
    does_exist = False
    for gate in unary_list:
        # gate = (q, t)

        q, t = gate
        if (q == q1 or q == q2):
            if (t1 < t and t < t2) or (t2 < t and t < t1):
                does_exist = True
                break
    
    return does_exist


def prune_annihilate(circuit):
    # n: number of qubits
    # circuit: [( ), ( ), ... ]
    # Unary gate: (q, t)
    # Binary (CZ) gate: (q1, q2, t)  # Assume that q1 < q2.

    unary_list = [gate for gate in circuit if len(gate) == 2]
    cz_list = [gate for gate in circuit if len(gate) == 3]

    keep_searching = True
    while keep_searching:
        annihilated_any = False
        num_cz = len(cz_list)
        for i in range(num_cz - 1):
            annihilated_in_j = False
            qi1, qi2, ti = cz_list[i]  # (qi1, qi2, ti)
            for j in range(i + 1, num_cz):
                qj1, qj2, tj = cz_list[j]  # (qj1, qj2, tj)
                if qi1 == qj1 and qi2 == qj2:
                    if not is_unary_between(qi1, qi2, ti, tj, unary_list):  # do pair-annihilate
                        cz_list = [cz_gate for cz_gate in cz_list if (cz_gate != (qi1, qi2, ti) and cz_gate != (qj1, qj2, tj))]
                        annihilated_any = True
                        annihilated_in_j = True
                        break
            if annihilated_in_j:
                break
        if not annihilated_any:
            keep_searching = False

    return unary_list + cz_list


def find_largest_leq(time_list, time):
    time_list_leq = [t for t in time_list if t <= time]

    return time_list_leq[-1]


def pre_find_largest_leq(unary_times, n, maximum_time):
    pre_found_largest_leq = [[] for _ in range(n)]

    for q in range(n):
        for time in range(maximum_time):
            q_times_unary = unary_times[q]
            pre_found_largest_leq[q].append(find_largest_leq(q_times_unary, time))

    return pre_found_largest_leq


def programming_params_general(gates_list, partition_list, n, num_partitions):  # for num_partitions >= 4
    # Input 1: gates_list [( ), ( ), ... ]
    # Unary gate: (q, t)
    # Binary (CZ) gate: (q1, q2, t)  # Assume that q1 < q2.
    # Input 2: partition_list [p_{0}, p_{1}, ... , p_{n - 1}]
    # Input 3: n (number of qubits)

    unary_times = [[] for _ in range(n)]
    for gate in gates_list:
        if len(gate) == 2:  # unary gate
            # qubit index = gate[0]
            # time = gate[1]
            unary_times[gate[0]].append(gate[1])
    for q in range(n):
        if unary_times[q]:
            unary_times[q].sort()
            if unary_times[q][0] != 0:
                unary_times[q].insert(0, 0)
        else:
            unary_times[q] = [0]
    
    migrations = []
    for q in range(n):
        for time in unary_times[q]:
            for p in range(num_partitions):
                if p != partition_list[q]:
                    migrations.append((q, time, p))
    
    cz_list = []
    for gate in gates_list:
        if len(gate) == 3:
            cz_list.append(gate)
    
    third_pairs = []
    for gate in cz_list:
        q, _q, time = gate
        t = find_largest_leq(unary_times[q], time)
        _t = find_largest_leq(unary_times[_q], time)

        for p in range(num_partitions):
            if p != partition_list[q] and p != partition_list[_q]:
                third_pairs.append(((q, t, p), (_q, _t, p)))

    num_migrations_single = len(migrations)
    num_migrations_pair = len(third_pairs)
    migrations += third_pairs

    num_cz = len(cz_list)
    num_migrations = len(migrations)
    A = np.zeros((num_cz, num_migrations))
    B = np.zeros((num_migrations_pair, num_migrations))

    for i in range(num_cz):
        gate = cz_list[i]  # (q1, q2, time)
        q1, q2, time = gate
        p1, p2 = partition_list[q1], partition_list[q2]
        for j in range(num_migrations):
            migration_any = migrations[j]
            if len(migration_any) == 3:  # (q, t, p)
                q, t, p = migration_any
                if q1 == q and find_largest_leq(unary_times[q1], time) == t and p2 == p:
                    A[i][j] = 1
                elif q2 == q and find_largest_leq(unary_times[q2], time) == t and p1 == p:
                    A[i][j] = 1
            else:  # ((q, t, p), (_q, _t, p))
                q, t, p = migration_any[0]
                _q, _t, p = migration_any[1]
                if q1 == q and q2 == _q:
                    if find_largest_leq(unary_times[q1], time) == t and find_largest_leq(unary_times[q2], time) == _t:
                        if p != p1 and p != p2:
                            A[i][j] = 1
    
    for i in range(num_migrations_pair):
        # a + b >= 2ab
        # a + b - 2ab >= 0

        B[i][num_migrations_single + i] = -2  # coefficient -2

        first_migration = third_pairs[i][0]
        second_migration = third_pairs[i][1]
        for j in range(num_migrations_single):
            if migrations[j] == first_migration or migrations[j] == second_migration:
                B[i][j] = 1
    
    C = np.concatenate((A, B), axis=0)

    v1 = np.ones((num_migrations_single, 1))
    v2 = np.zeros((num_migrations_pair, 1))
    v = np.concatenate((v1, v2), axis=0)

    u1 = np.ones((num_cz, 1))
    u2 = np.zeros((num_migrations_pair, 1))
    u = np.concatenate((u1, u2), axis=0)

    return v, C, u  # minimize v^T * x subject to C * x >= u


def random_circuit(n=50, gates_per_qubit=50, binary_percent=50):
    qubits = list(range(n))

    if binary_percent == 50:
        num_cz = int(n * gates_per_qubit / 4)
        num_unary = num_cz * 2
        num_gates = num_unary + num_cz

        times = list(range(num_gates))
        cz_times = sorted(random.sample(times, num_cz))
        unary_times = [t for t in times if t not in cz_times]

        circuit = [None for _ in range(num_gates)]
        for t in unary_times:
            q = random.choice(qubits)
            circuit[t] = (q, t)
        for t in cz_times:
            two_qubits = sorted(random.sample(qubits, 2))
            q1, q2 = two_qubits[0], two_qubits[1]
            circuit[t] = (q1, q2, t)
    else:
        binary_fraction = binary_percent / 100
        num_cz = int(n * gates_per_qubit * binary_fraction / 2)
        num_unary = int(num_cz * 2 * (1 / binary_fraction - 1))
        num_gates = num_unary + num_cz

        times = list(range(num_gates))
        cz_times = sorted(random.sample(times, num_cz))
        unary_times = [t for t in times if t not in cz_times]

        circuit = [None for _ in range(num_gates)]
        for t in unary_times:
            q = random.choice(qubits)
            circuit[t] = (q, t)
        for t in cz_times:
            two_qubits = sorted(random.sample(qubits, 2))
            q1, q2 = two_qubits[0], two_qubits[1]
            circuit[t] = (q1, q2, t)
    
    return circuit


def random_circuit_reference(n=50, gates_per_qubit=50, binary_percent=50):
    qubits = list(range(n))
    num_layers = gates_per_qubit

    gates_without_time = []

    p = binary_percent / 100

    for layer in range(num_layers):
        unary_qubits = []
        for q in range(n):
            random_float = random.random()
            if random_float < 1 - p:
                unary_qubits.append(q)
        if unary_qubits and binary_percent <= 60:
            unary_qubits.pop()

        if binary_percent == 100:
            unary_qubits = []

        for u_qubit in unary_qubits:
            gates_without_time.append((u_qubit,))

        binary_qubits = [q for q in qubits if q not in unary_qubits]
        random.shuffle(binary_qubits)
        if len(binary_qubits) % 2 != 0:
            binary_qubits.pop()
        gates_without_time += [(binary_qubits[i], binary_qubits[i + 1]) for i in range(0, len(binary_qubits), 2)]
    
    gates_with_time = []
    t = 0
    for gate_candidate in gates_without_time:
        gates_with_time.append(gate_candidate + (t,))
        t += 1
    
    return gates_with_time


def refined_weights(circuit, n):
    # edge_weights = [(v1, v2, w), ... ], where (v1, v2) are lexicographically ordered.
    # n: number of qubits

    edge_weights = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            circuit_ij = [gate for gate in circuit if (len(gate) == 2 and gate[0] in (i, j)) or (len(gate) == 3 and gate[:2] == (i, j))]
            circuit_ij_unary = [gate for gate in circuit_ij if len(gate) == 2]
            circuit_ij_cz = [gate for gate in circuit_ij if len(gate) == 3]
            w = 0
            while circuit_ij_cz:
                if len(circuit_ij_cz) == 1:
                    w += 1
                    circuit_ij_cz.clear()
                else:  # len(circuit_ij_cz) >= 2
                    t1, t2 = circuit_ij_cz[0][2], circuit_ij_cz[1][2]
                    between_for_i = False
                    between_for_j = False
                    for unary_gate in circuit_ij_unary:  # (q, t)
                        if unary_gate[0] == i and t1 < unary_gate[1] and unary_gate[1] < t2:
                            between_for_i = True
                        if unary_gate[0] == j and t1 < unary_gate[1] and unary_gate[1] < t2:
                            between_for_j = True
                    if between_for_i and between_for_j:
                        w += 1
                        circuit_ij_cz.pop(0)
                    elif not between_for_i and between_for_j:  # migrate i to partition of j
                        w += 1
                        first_block_unary_time = 100000
                        for unary_gate in circuit_ij_unary:
                            if unary_gate[0] == i and unary_gate[1] > t2:
                                first_block_unary_time = unary_gate[1]
                                break
                        while circuit_ij_cz:
                            if circuit_ij_cz[0][2] < first_block_unary_time:
                                circuit_ij_cz.pop(0)
                            else:
                                break
                    else:
                        w += 1
                        first_block_unary_time = 100000
                        for unary_gate in circuit_ij_unary:
                            if unary_gate[0] == j and unary_gate[1] > t2:
                                first_block_unary_time = unary_gate[1]
                                break
                        while circuit_ij_cz:
                            if circuit_ij_cz[0][2] < first_block_unary_time:
                                circuit_ij_cz.pop(0)
                            else:
                                break
            edge_weights.append((i, j, w))

    return edge_weights


def hyperedges_weights(refined_weights_list):
    # refined_weights_list = [(q1, q2, w), ... ]

    hyperedges = []
    edge_weights = []
    for triple in refined_weights_list:
        if triple[2] != 0:
            hyperedges.append(triple[0])
            hyperedges.append(triple[1])
            edge_weights.append(triple[2])
    
    return hyperedges, edge_weights


def gen_migrations(gates_list, partition_list, n, num_partitions):
    unary_times = [[] for _ in range(n)]
    for gate in gates_list:
        if len(gate) == 2:  # unary gate
            unary_times[gate[0]].append(gate[1])

    for q in range(n):
        if unary_times[q]:
            unary_times[q].sort()
            if unary_times[q][0] != 0:
                unary_times[q].insert(0, 0)
        else:
            unary_times[q] = [0]

    migrations = []
    for q in range(n):
        for time in unary_times[q]:
            for p in range(num_partitions):
                if p != partition_list[q]:
                    migrations.append((q, time, p))
    # migrations is a list like [(q, time, p), ... ].
    
    return migrations


def covered_gates_density(migrations_set, unary_times, remaining_cz, partition_list, pre_found_largest_leq):
    p = migrations_set[0][2]

    num_covered = 0
    for cz_gate in remaining_cz:
        this_covered = False
        q1, q2, t = cz_gate

        if partition_list[q1] == p:
            if (q2, pre_found_largest_leq[q2][t], p) in migrations_set:
                this_covered = True
        if partition_list[q2] == p:
            if (q1, pre_found_largest_leq[q1][t], p) in migrations_set:
                this_covered = True
        desired_1 = (q1, pre_found_largest_leq[q1][t], p)
        desired_2 = (q2, pre_found_largest_leq[q2][t], p)
        if (desired_1 in migrations_set) and (desired_2 in migrations_set):
            this_covered = True
        
        if this_covered:
            num_covered += 1
    
    return num_covered / len(migrations_set)


def simple_greedy(subset, unary_times, remaining_cz, partition_list, pre_found_largest_leq):
    if len(subset) == 1:
        current_density = covered_gates_density(subset, unary_times, remaining_cz, partition_list, pre_found_largest_leq)
        if current_density > 0:
            return subset, current_density
        if current_density == 0:
            return [], current_density
    
    max_density = covered_gates_density(subset, unary_times, remaining_cz, partition_list, pre_found_largest_leq)
    if max_density == 0:
        return [], max_density
    
    delete_best = None
    for delete_single in subset:
        one_less_subset = [migration for migration in subset if migration != delete_single]
        current_density = covered_gates_density(one_less_subset, unary_times, remaining_cz, partition_list, pre_found_largest_leq)
        if current_density > max_density:
            delete_best = delete_single
            max_density = current_density
    
    if delete_best:
        one_less_best = [migration for migration in subset if migration != delete_best]
        return simple_greedy(one_less_best, unary_times, remaining_cz, partition_list, pre_found_largest_leq)
    else:
        return subset, max_density


def delete_covered(cz_list, unary_times, subset, partition_list, pre_found_largest_leq):
    current_partition = subset[0][2]

    cz_list_still_left = []
    for cz_gate in cz_list:
        still_not_covered = True
        q, _q, time = cz_gate
        t = pre_found_largest_leq[q][time]
        _t = pre_found_largest_leq[_q][time]

        if partition_list[_q] == current_partition:
            if (q, t, partition_list[_q]) in subset:
                still_not_covered = False
        elif partition_list[q] == current_partition:
            if (_q, _t, partition_list[q]) in subset:
                still_not_covered = False
        else:
            if (q, t, current_partition) in subset and (_q, _t, current_partition) in subset:
                still_not_covered = False

        if still_not_covered:
            cz_list_still_left.append(cz_gate)
    
    return cz_list_still_left


def G_star(circuit_left, migrations_left, migrations_selected, partition_list, n, num_partitions, unary_times, unary_list, pre_found_largest_leq):
    print("G_star iteration")
    remaining_cz = [gate for gate in circuit_left if len(gate) == 3]  # CZ gates list

    same_partition_subsets = [[] for _ in range(num_partitions)]
    for mig in migrations_left:
        same_partition_subsets[mig[2]].append(mig)

    most_dense_among_partitions = None
    best_density_among_partitions = -1
    for partition in range(num_partitions):
        partition_migrations = same_partition_subsets[partition]
        densest_subset = partition_migrations
        if densest_subset:
            densest_subset, max_density_within_partition = simple_greedy(densest_subset, unary_times, remaining_cz, partition_list, pre_found_largest_leq)
            if max_density_within_partition > best_density_among_partitions:
                most_dense_among_partitions = densest_subset
                best_density_among_partitions = max_density_within_partition
    
    migrations_selected += most_dense_among_partitions
    migrations_left = [migration for migration in migrations_left if migration not in most_dense_among_partitions]

    cz_list_still_left = delete_covered(remaining_cz, unary_times, most_dense_among_partitions, partition_list, pre_found_largest_leq)
    
    if len(cz_list_still_left) == 0:
        return migrations_selected
    else:
        circuit_left = unary_list + cz_list_still_left
        return G_star(circuit_left, migrations_left, migrations_selected, partition_list, n, num_partitions, unary_times, unary_list, pre_found_largest_leq)


def prune_home_gates(gates_list, partition_list):
    pruned_gates_list = []
    for gate in gates_list:
        if len(gate) == 2:  # unary gate
            pruned_gates_list.append(gate)
        else:  # CZ gate
            q1 = gate[0]
            q2 = gate[1]
            if partition_list[q1] != partition_list[q2]:
                pruned_gates_list.append(gate)
    
    return pruned_gates_list


def solve_with_gurobi(v, C, u):
    # minimize v^T * x subject to C * x >= u

    num_variables = len(v)
    num_constraints = len(C)

    opt_mod = Model(name="linear program")

    constraints = []

    for j in range(num_variables):
        new_var = opt_mod.addVar(name = str(j), vtype = GRB.BINARY)
        if j == 0:
            obj_fn = new_var * int(v[j][0])
            for i in range(num_constraints):
                constraints.append(new_var * C[i][j])
        else:
            obj_fn += (new_var * int(v[j][0]))
            for i in range(num_constraints):
                constraints[i] += (new_var * C[i][j])
    
    opt_mod.setObjective(obj_fn, GRB.MINIMIZE)

    for i in range(num_constraints):
        opt_mod.addConstr(constraints[i] >= int(u[i][0]), name = 'c' + str(i))
    
    opt_mod.optimize()
    opt_mod.write("linear_model_running_example.lp")

    print('Objective Function Value: %f' % opt_mod.objVal)

    solution = []
    for v in opt_mod.getVars():
        solution.append(int(v.x))

    return solution


def martinez(circuit, n, k):
    unary_times = [[] for _ in range(n)]
    for gate in circuit:
        if len(gate) == 2:
            unary_times[gate[0]].append(gate[1])

    for q in range(n):
        if unary_times[q]:
            unary_times[q].sort()
            if unary_times[q][0] != 0:
                unary_times[q].insert(0, 0)
        else:
            unary_times[q] = [0]
    for q in range(n):
        unary_times[q].append(100000)
    
    cz_gates = [gate for gate in circuit if len(gate) == 3]
    
    node_weights = [10000] * n + [1] * len(cz_gates)
    num_nodes = len(node_weights)

    hyperedges = []
    hyperedges_for_test = []
    hyperedge_indices = [0]
    edge_weights = []
    current_hyper_idx = 0
    for q in range(n):
        hyperedge_qubit = [q]
        for u_time_idx in range(len(unary_times[q]) - 1):
            u_time_now, u_time_next = unary_times[q][u_time_idx], unary_times[q][u_time_idx + 1]
            q_and_u_time = []
            for cz_idx in range(len(cz_gates)):
                cz = cz_gates[cz_idx]
                if (cz[0] == q or cz[1] == q) and cz[2] >= u_time_now and cz[2] < u_time_next:
                    q_and_u_time.append(n + cz_idx)
            if q_and_u_time:
                edge = hyperedge_qubit + q_and_u_time
                hyperedges += edge
                hyperedges_for_test.append(edge)
                current_hyper_idx += len(edge)
                hyperedge_indices.append(current_hyper_idx)
                edge_weights.append(1)
    num_nets = len(edge_weights)

    hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)
    hypergraph.printGraphState()

    context = kahypar.Context()
    context.loadINIconfiguration("config/km1_kKaHyPar_sea20.ini")

    context.setK(k)
    context.setEpsilon(0.1)  # default to 0.1

    kahypar.partition(hypergraph, context)

    # hypergraph.printGraphState()
    partition_list = [hypergraph.blockID(i) for i in range(n)]
    print("Partition: ", partition_list)
    print()
    print("Groups:")
    for i in range(k):
        for j in range(n):
            if hypergraph.blockID(j) == i:
                print(j, end=" ")
        print()
    print()
    num_global = 0
    for edge in hyperedges_for_test:
        num_global += (len(set([hypergraph.blockID(node) for node in edge])) - 1)
    print("Cut value: ", num_global)
    print()

    return num_global, partition_list, hypergraph
