import networkx as nx
import matplotlib.pyplot as plt
import random
from itertools import combinations
from collections import deque

import numpy
import numpy as np
import scipy


def prob(p):
    if random.random() < p:
        return True
    else:
        return False


G = nx.Graph()
vertices = [i for i in range(0, 20)]
k_list = [i for i in range(5, 101, 5)]
G.add_nodes_from(vertices)


def connection(G, s):
    Q = deque([])
    dist = [True] + [False for i in range(2, 21)]
    Q.append(s)
    while len(Q) != 0:
        cur = Q.popleft()
        for neibor in  G.neighbors(cur):
                if not (dist[neibor]):
                    dist[neibor] = True
                    Q.append(neibor)
    #checking connection
    for i in dist:
        if not i:
            return False
    return True



set_of_edges = list(combinations(vertices, 2))
counter = [0 for i in range(len(k_list))]
for k in range(len(k_list)):
    q = k_list[k] / 100
    C_q = 0
    print("the probability is: ", q)
    for iteraition in range(1, 11):
        for edge in set_of_edges:
            if prob(q):
                G.add_edges_from([edge])
        if connection(G, 1):
            C_q +=1
        edges = G.edges()
        G.remove_edges_from(edges)
    print(C_q)
    counter[k] = C_q*10

#plotting graph

q_list = [i/100 for i in range(5, 101, 5)]
plt.plot(q_list, counter, "-*r")
plt.xlabel("q ")
plt.ylabel("C(q) connectivity (%)")
plt.title(" grapg c(q) as function of q")
plt.show()
print("end")




#question 2

n_list = [i for i in range(5, 201, 5)]
q = 0.5
t=0

def BFS(G, s, n):
    Q = deque([])
    dist = [10000 for i in range(n)]
    dist[s] = 0
    Q.append(s)
    while len(Q) != 0:
        cur = Q.popleft()
        for neibor in G.neighbors(cur):
            if dist[neibor] == 10000:
                dist[neibor] = dist[cur] + 1
                Q.append(neibor)
    return dist

average_t = []
for n in range(5,201,5):

    sum =0
    for iterait in range(10):
        G = nx.Graph()

        ver = [i for i in range(n)]
        G.add_nodes_from(ver)
        set_of_edges = list(combinations(ver, 2))
        for edge in set_of_edges:
            if prob(q):
                G.add_edges_from([edge])
        #after creating G

        mat = nx.to_numpy_array(G)
        A = np.linalg.matrix_power(mat, 3)   #A^3
        #nx.draw_circular(G)
        #plt.show()
        triangle_counter = 0

        sum += A.trace() // 6

    average_t.append(sum/10)#we change to sum result from sum





plt.plot(n_list, average_t, "-*r")
plt.xlabel("n - verticies")
plt.ylabel("t3(n)")
plt.title("t3(n) as function of n")
plt.show()
print("end")


#QUESTION 3

#straight line
G_strait = nx.Graph()
G_circle = nx.Graph()
v = [i for i in range(20)]
e = [(i, (i+1)) for i in range(19)]
e_c = e + [(v[19], v[0])]


#initialize base graph
random.shuffle(v)
G_strait.add_nodes_from(v)

e = [(v[i], v[i+1]) for i in range(19)]
G_strait.add_edges_from(e)

G_circle.add_nodes_from(v)
G_circle.add_edges_from(e_c)

G_strait=nx.path_graph(20)
#G_circle = G_strait
#G_circle.add_edges_from([(G_strait.nodes(), G_strait.nodes()[19])])


def accention(G,name):
    k_list = [i for i in range(5, 101, 5)]
    set_of_edges = list(combinations(vertices, 2))
    accent = []
    e_q = []
    data_e_q = []
    edge_remove = []
    for k in range(len(k_list)):
        q = k_list[k] / 100
        print("the probability is: ", q)
        for iteraition in range(1, 11):
            for edge in set_of_edges:
                if prob(q) and edge not in G.edges():
                    G.add_edges_from([edge])
                    edge_remove.append(edge)
            for j in range(20):
                accent.append(max(BFS(G, j, 20)))
            e_q_G = numpy.average(accent)  # average accention in single graph
            accent = []
            e_q.append(e_q_G)

            G.remove_edges_from(edge_remove)
            edge_remove = []
        data_e_q.append(numpy.average(e_q)) # average accention of all avr accention  graph

    q_list = [i/100 for i in range(5, 101, 5)]
    plt.plot(q_list, data_e_q, "-*r")
    plt.xlabel("probability P")
    plt.ylabel("e(q)")
    plt.title(name + "base garah: e(q) as funqion of q")
    plt.show()


accention(G_strait,"staight")
accention(G_circle,"circle")

