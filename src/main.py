import utils
from utils import *
import random
import time
import kahypar as kahypar
import sys
import time
import utils_volume
from utils_volume import *

# python3 main.py

# ----------------------------------------------------------------------------------------------------
''' Random Circuit, Solve End-to-End with Gurobi only '''
# Hyperparameters
n = 50  # number of qubits
gates_per_qubit = 50
binary_percent = 10
k = 10  # number of partitions
running_ex_partition = False

# circuit = random_circuit(n, gates_per_qubit)
# circuit = random_circuit(n, gates_per_qubit, binary_percent)
circuit = random_circuit_reference(n, gates_per_qubit, binary_percent)

print("  Circuit before pruning: ", circuit)
print("num_gates before pruning: ", len(circuit))
circuit = prune_annihilate(circuit)
print("   Circuit after pruning: ", circuit)
print(" num_gates after pruning: ", len(circuit))
refined_weights_list = refined_weights(circuit, n)
print("Refined weights: ", refined_weights_list)
hyperedges, edge_weights = hyperedges_weights(refined_weights_list)
node_weights = [1] * n
hyperedge_indices = [2 * i for i in range(len(edge_weights) + 1)]
print("Hyperedges: ", hyperedges)
print("Hyperedge indices: ", hyperedge_indices)
print("Edge weights: ", edge_weights)
num_nodes = n
num_nets = len(edge_weights)  # number of hyperedges

hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, k, edge_weights, node_weights)
hypergraph.printGraphState()

context = kahypar.Context()
context.loadINIconfiguration("config/km1_kKaHyPar_sea20.ini")

context.setK(k)
context.setEpsilon(0.1)  # default to 0.1

kahypar.partition(hypergraph, context)

# hypergraph.printGraphState()
partition_list = [hypergraph.blockID(i) for i in range(num_nodes)]
print("Partition: ", partition_list)
print("Groups:")
for i in range(k):
    for j in range(num_nodes):
        if hypergraph.blockID(j) == i:
            print(j, end="")

if running_ex_partition:
    print("Change partition to running example.")
    partition_list = [0, 1, 1, 2, 2]

home_cz_pruned_circuit = prune_home_gates(circuit, partition_list)
print("Home CZ gates deleted: ", home_cz_pruned_circuit)
print("Number of gates after home CZ gates deleted: ", len(home_cz_pruned_circuit))
circuit = home_cz_pruned_circuit

migrations = gen_migrations(circuit, partition_list, n, k)  # unary gates: (0, 2), (1, 5), (2, 4), (4, 6)
print("Migrations: ", migrations)
print("Number of all possible migrations: ", len(migrations))

# minimize v^T * x subject to C * x >= u
v, C, u = programming_params_general(circuit, partition_list, n, k)
# print("v transposed: ", v.T)
# print("u transposed: ", u.T)
# print("Matrix C:")
# for C_row in C:
#     print([int(binary) for binary in C_row])
solution_extended = solve_with_gurobi(v, C, u)
solution = solution_extended[:len(migrations)]
print("Extended solution: ", solution_extended)
selected_migrations = [migrations[i] for i in range(len(migrations)) if solution[i] == 1]
print("Selected migrations: ", selected_migrations)
print("Number of migrations: ", len(selected_migrations))
# ----------------------------------------------------------------------------------------------------
