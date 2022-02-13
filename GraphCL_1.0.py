# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
import time
import networkx as nx
import numpy as np
import torch
import torch.optim as optim


# %%
# from torch_geometric.datasets import TUDataset
# from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader
# import torch_geometric.transforms as T


# %%
import os
import re
import numpy as np
import networkx as nx
from collections import Counter


# %%
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp


# %%
from random import choice
from heapq import nsmallest
from itertools import product
try:
    from collections import Counter
except ImportError: # Counter is not available in Python before v2.7
    from recipe_576611_1 import Counter
try:
    from joblib import Parallel, delayed
except ImportError:
    pass
import io

# method that returns true iff only one element of the container is True
def unique(container):
    return Counter(container).get(True, 0) == 1


# the Node representation of the data
class Node:
    # data is an array of values
    def __init__(self, data=None, node_id =None):
        self.data = data
        self.children = {}      # dict mapping level and children
        self.parent = None
        self.node_id = node_id

    # addChild adds a child to a particular Node and a given level i
    def addChild(self, child, i):
        try:
            # in case i is not in self.children yet
            if(child not in self.children[i]):
                self.children[i].append(child)
        except(KeyError):
            self.children[i] = [child]
        child.parent = self

    # getChildren gets the children of a Node at a particular level
    def getChildren(self, level):
        retLst = [self]
        try:
            retLst.extend(self.children[level])
        except(KeyError):
            pass
        
        return retLst

    # like getChildren but does not return the parent
    def getOnlyChildren(self, level):
        try:
            return self.children[level]
        except(KeyError):
            pass
        
        return []

    #
    # Overview: get sum pooling of all the children of a node present at a level 
    #
    # Input: node and the level @ which the node is present
    #
    # Output: sum pooled np array - ball embedding of the ball represented by this node
    #
    def get_all_children(self, level, minlevel):
        ball_children = {level : [(self, self.node_id)]}
        #ball_children = {}
        #feat_dim = self.data.shape[-1]
        #ball_data = []
        for lev in reversed(range(minlevel, level)):
            if (lev in self.children.keys()):
                ball_children[lev]=[]
                for element in self.children[lev]:
                    ball_children[lev].append((element, element.node_id))
        #ball_data.append(np.sum([node.data for node in ball_children], 0))

        #return np.array(ball_data).reshape(-1, 1 , feat_dim)
        return ball_children



    def removeConnections(self, level):
        if(self.parent != None):
            self.parent.children[level+1].remove(self)
            self.parent = None

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)


class CoverTree:
    
    #
    # Overview: initalization method
    #
    # Input: distance function, root, maxlevel, minlevel, base, and
    #  for parallel support jobs and min_len_parallel. Here root is a
    #  point, maxlevel is the largest number that we care about
    #  (e.g. base^(maxlevel) should be our maximum number), just as
    #  base^(minlevel) should be the minimum distance between Nodes.
    #
    #  In case parallel is enbaled (jobs > 1), min_len_parallel is the
    #  minimum number of elements at a given level to have their
    #  distances to the element to insert or query evaluated.
    #
    def __init__(self, distance, maxlevel, root = None, base = 2,
                 jobs = 1, min_len_parallel = 100):
        self.distance = distance
        self.root = root
        self.maxlevel = maxlevel
        self.minlevel = maxlevel # the minlevel will adjust automatically
        self.base = base
        self.jobs = jobs
        self.min_len_parallel = min_len_parallel
        #the ct_dic stores level -> ball representations at thet level -> list of nodes for every ball representation
        self.ct_dict = {}
        # for printDotty
        self.__printHash__ = set()
    #
    # Overview: insert an element p into the tree
    #
    # Input: p
    # Output: nothing
    #
    
    
    #
    # Overview: insert an element p into the tree
    #
    # Input: p
    # Output: nothing
    #
    def insert(self, p, node_id):
        if self.root == None:
            self.root = Node(p, node_id)
        else:
            self.insert_iter(p, node_id)

            
    #
    # Overview: behaves like knn(p, k) and insert(p). This method
    # exists for efficiency reason
    #
    # Input: point p, and k the number of nearest neighbors to return
    #
    # Output: Nearest points with respect to the distance metric
    #          self.distance() and optionally their distances
    #
    def knn_insert(self, k, p, without_distance = False):
        if self.root == None:
            self.root = Node(p)
            return []
        else:
            return self._result_(self.knn_insert_iter(k, p), without_distance)
    
    #
    # Overview: get the k-nearest neighbors and their distances of an element
    #
    # Input: point p, and k the number of nearest neighbors to return
    #
    # Output: Nearest points with respect to the distance metric
    #          self.distance() and optionally their distances
    #
    def knn(self, k, p, without_distance = False):
        if self.root == None:
            return []
        else:
            return self._result_(self.knn_iter(k, p), without_distance)

    #
    # Overview: find an element in the tree
    #
    # Input: Node p
    # Output: True if p is found False otherwise
    #
    def find(self, p):
        return self.distance(self.knn(1, p, True)[0], p) == 0


    # Overview:insert an element p in to the cover tree
    #
    # Input: point p
    #
    # Output: nothing
    #
    def insert_iter(self, p, node_id):
        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        i = self.maxlevel
        used = 0
        while True:
            # get the children of the current level
            # and the distance of the all children
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            d_p_Q = self._min_ds_(Q_p_ds)

            if d_p_Q == 0.0 :    # already there, no need to insert
                print("\n",node_id," already there")
                return
            elif d_p_Q > self.base**i :
                break
            # or i==-5: # the found parent should be right
                # print(d_p_Q, "i is :", i)
                # or i==-6 )and used == 1)
                
            else: # d_p_Q <= self.base**i, keep iterating

                # find parent
                if self._min_ds_(Qi_p_ds) <= self.base**i:
                    parent = choice([q for q, d in Qi_p_ds if d <= self.base**i])
                    pi = i
                    used = 1
                
                # construct Q_i-1
                Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= self.base**i]
                i -= 1
        # insert p
        parent.addChild(Node(p, node_id), pi)
        # update self.minlevel
        self.minlevel = min(self.minlevel, pi)
        
    # Overview:get the nearest neighbor, iterative
    #
    # Input: query point p
    #
    # Output: the nearest Node 
    #
    def knn_iter(self, k, p):
        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        for i in reversed(range(self.minlevel, self.maxlevel + 1)):
            # get the children of the current Qi_p_ds and
            # the best distance at the same time
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            _, d_p_Q = self._kmin_p_ds_(k, Q_p_ds)[-1]

            #create the next set
            Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= d_p_Q + self.base**i]

        #find the minimum
        return self._kmin_p_ds_(k, Qi_p_ds)


    #
    # Overview: query the k-nearest points from p and then insert p in
    # to the cover tree (at no additional cost)
    #
    # Input: point p
    #
    # Output: nothing
    #
    def knn_insert_iter(self, k, p):
        Qi_p_ds = [(self.root, self.distance(p, self.root.data))]
        i = self.maxlevel
        found_parent = False
        already_there = False
        while (not already_there and not found_parent) or i >= self.minlevel:
            # get the children of the current level
            # and the distance of all children
            Q_p_ds = self._getChildrenDist_(p, Qi_p_ds, i)
            d_k = self._kmin_p_ds_(k, Q_p_ds)
            _, d_p_Q_h = d_k[-1]
            _, d_p_Q_l = d_k[0]

            if d_p_Q_l == 0.0:    # already there, no need to insert
                already_there = True
            elif not already_there and  not found_parent and d_p_Q_l > self.base**(i-1):
                found_parent = True
                
            # remember potential parent
            if self._min_ds_(Qi_p_ds) <= self.base**i:
                parent = choice([q for q, d in Qi_p_ds if d <= self.base**i])
                pi = i

            # construct Q_i-1
            Qi_p_ds = [(q, d) for q, d in Q_p_ds if d <= d_p_Q_h + self.base**i]
            i -= 1

        # insert p
        if not already_there and found_parent:
            parent.addChild(Node(p), pi)
            # update self.minlevel
            self.minlevel = min(self.minlevel, pi)
        
        # find the minimum
        return self._kmin_p_ds_(k, Qi_p_ds)
    

        
    #
    # Overview: creates ball_representations of the cover tree
    #
    # Input: cover tree object
    #
    # Output: a dictionary with cover tree in terms of level -> ball_reps @ that level - nodes covered by that ball rep
    #
    def make_ball_rep(self):
        current = self.root
        nodes = [current]
        x = set()
        for i in range(188):
            x.add(i)
        for level in reversed(range(self.minlevel, self.maxlevel+1)):
            self.ct_dict[level] = {}
        self.ct_dict[self.maxlevel][(current, current.node_id)]=[(current, current.node_id)]
        while(len(nodes)):
            for key in current.children.keys():
                nodes.extend(current.children[key])
                
            for key in current.children.keys():
                if (current, current.node_id) not in self.ct_dict[key].keys():
                    self.ct_dict[key][(current, current.node_id)]=[]
            for key in current.children.keys():  
                for element in current.children[key]:
                    x = x - {element.node_id}
                    self.ct_dict[key][(current, current.node_id)].append((element, element.node_id))
            #assume x nodes are kids of root at level with index maxlevel
            nodes.pop(0)
            if(len(nodes)):
                current = nodes[0]

        no = 0
        for l in self.ct_dict.keys():
            for br in self.ct_dict[l].keys():
                no += len(self.ct_dict[l][br])
        #ct_dict made
        for level in reversed(range(self.minlevel, self.maxlevel)):

            for ball_rep in self.ct_dict[level].keys():
                temp_dic = {}
                for ele in self.ct_dict[level][ball_rep]:
                    temp_dic = ele[0].get_all_children(level, self.minlevel)
                    if(bool(temp_dic)):
                        for key in temp_dic.keys():
                            if key!= level :
                                if ball_rep in self.ct_dict[key].keys():
                                    self.ct_dict[key][ball_rep].extend(temp_dic[key])
                                else :
                                    self.ct_dict[key][ball_rep]= temp_dic[key]

        root_kids=0
        for level in self.ct_dict.keys():
            for ball_rep in self.ct_dict[level].keys():
                if(ball_rep[1] == self.root.node_id):
                    root_kids += len(self.ct_dict[level][ball_rep])
        for lev in reversed(range(self.minlevel+1,self.maxlevel+1)):
            for ball_rep in self.ct_dict[lev]:
                for l in range(self.minlevel, lev):
                    if ball_rep in self.ct_dict[l].keys():
                        self.ct_dict[lev][ball_rep].extend(self.ct_dict[l][ball_rep])

        return self.ct_dict, root_kids, no, self.maxlevel, self.minlevel
    #
    # Overview: gets hard negatives given a point p
    #
    # Input: cover tree object and p whose hard negatives are needed
    #
    # Output: a list of hard negative points
    #
    def get_hard_negatives(self , p):
        found = 0
        p_ball_rep = p
        p_level = -1
        h_n = []
        nn = []
        neg=[]
        no_hn = 0
        for level in reversed(range(self.minlevel, self.maxlevel+1)):
            for ball_rep in self.ct_dict[level].keys():
                for x in self.ct_dict[level][ball_rep]:
                    comp1 = x[0].data == p.data
                    if comp1.all():
                        p_level = level
                        p_ball_rep = ball_rep
                        found = 1
                        break
                if(found == 1):
                    break
            if(found == 1):
                break
        #adding negatives from the level of the node to max level
        #for level in reversed(range(p_level,self.maxlevel)):
        list_dis=[]
        for ball_rep in self.ct_dict[p_level].keys():
            if(ball_rep[1] != p_ball_rep[1]):
                dis = self.distance(p.data, ball_rep[0].data)
                list_dis.append((ball_rep, dis))
        #adding hard negatives
        if(len(list_dis)):
                min_d, idx = min((list_dis[i][1], i) for i in range(len(list_dis)))
                h_n.extend(self.ct_dict[p_level][list_dis[idx][0]])
        #adding normal negatives
        for i in range(len(list_dis)):
                if (i != idx ):
                    nn.extend(self.ct_dict[p_level][list_dis[i][0]])
        #adding negatives from after node's level to minlevel
        for level in reversed(range(self.minlevel, p_level)):
            #list of tuples -> (ball_rep , dist_with_p)
            list_dis = []
            for ball_rep in self.ct_dict[level].keys():
                #not needed - (ball_rep != p_ball_rep) and 
                if(not(ball_rep[1]==p.node_id)):
                    dis = self.distance(p.data, ball_rep[0].data)
                    list_dis.append((ball_rep, dis))
            #adding hard negatives
            if(len(list_dis)):
                min_d, idx = min((list_dis[i][1], i) for i in range(len(list_dis)))
                h_n.extend(self.ct_dict[level][list_dis[idx][0]])
            #adding normal negatives
            for i in range(len(list_dis)):
                if (i != idx ):
                    nn.extend(self.ct_dict[level][list_dis[i][0]])
      
        for element in range(len(h_n)):
            no_hn += 1
            neg.append(h_n[element])
        for element in range(len(nn)):
            neg.append(nn[element])
        return no_hn, neg

    #
    # Overview: get negatives given a node
    #
    # Input: cover tree object
    #New neg
    def get_negatives(self , p):
        found = 0
        p_ball_rep = p
        p_level = -1
        h_n = {}
        nn = {}
        neg= {}
        no_hn = 0
        for level in reversed(range(self.minlevel, self.maxlevel)):
            for ball_rep in self.ct_dict[level].keys():
                for x in self.ct_dict[level][ball_rep]:
                    comp1 = x[0].data == p.data
                    if comp1.all() or p.node_id==ball_rep[1]:
                        #level @ which p's ball representative is
                        p_level = level
                        p_ball_rep = ball_rep
                        found = 1
                        break
                if(found == 1):
                    break
            if(found == 1):
                break
        #adding negatives from the level of the node to min level
        #for level in reversed(range(p_level,self.maxlevel)):
        list_dis=[]
        for ball_rep in self.ct_dict[p_level].keys():
            if(ball_rep != p_ball_rep):
                dis = self.distance(p.data, ball_rep[0].data)
                list_dis.append((ball_rep, dis))
        #adding hard negatives
        if(len(list_dis)):
                min_d, idx = min((list_dis[i][1], i) for i in range(len(list_dis)))
                h_n = list_dis[idx][0][0].get_all_children(p_level, self.minlevel)
        #adding normal negatives
        for i in range(len(list_dis)):
                #some scope here
                if (i != idx ):
                    nn=list_dis[i][0][0].get_all_children(p_level, self.minlevel)
                    for k in nn.keys():
                        if (k in neg.keys()):
                            neg[k].extend(nn[k])
                        else :
                            neg[k] = nn[k]
        
        return h_n, neg

    # Output: a list of np array(ball embeddings) of each ball in the cover tree
    #
    # def get_ball_embeddings(self):
    #     ball_embeddings = []
    #     for level in self.ct_dict.keys():
    #         for ball_rep in self.ct_dict[level].keys():
    #             data_list = []
    #             data_list.append(ball_rep.data)
    #             for x in self.ct_dict[level][ball_rep]:
    #                 data_list.append(x.data)
    #             ball_embeddings.append(np.sum(data_list, 0))

    #     return ball_embeddings

    def get_pos_neg(self, p, level):
        #get positive, list(hard negatives), list(negatives) of node p @ level
        found = 0
        #pos node
        pos =(p, p.node_id)
        #ball_rep of the sent node p
        p_ball_rep = (p, p.node_id)
        for ball_rep in self.ct_dict[level].keys():
            for element in self.ct_dict[level][ball_rep]:
                if p.node_id == element[1]:
                    p_ball_rep = ball_rep
                    found =1
                    break
            if(found==1):
                break

        #positive from the level
        list_pdis = []
        if found == 1 :
            for element in self.ct_dict[level][p_ball_rep]:
                if element[1] != p.node_id :
                    dis = self.distance(p.data, element[0].data)
                    list_pdis.append((element[0], dis))
            if(len(list_pdis)):
                min_d, idx = min((list_pdis[i][1], i) for i in range(len(list_pdis)))
                pos = (list_pdis[idx][0], list_pdis[idx][0].node_id)
            else :
                results = self.knn(2, p.data, True)
                pos = results[1]
        else:
            results = self.knn(2, p.data, True)
            pos = results[1]
        #pos looks fine till here

        # now negatives from the level
        all_neg=[]
        for ball_rep in self.ct_dict[level].keys():
            if ball_rep[1] != p_ball_rep[1]:
                dis = self.distance(p.data, ball_rep[0].data)
                all_neg.append((ball_rep[0], dis))
        #so all_neg rn has (ball_rep node, distance) 
        all_neg.sort(key = lambda x : x[1])
        n_neg=[]
        o_neg = []
        temp=[]
        hard_neg=[]
        if(len(all_neg)==0):
            # print("no other ball rep @ this level")
            # while(len(all_neg)==0 and level < self.maxlevel-1):
            #     level = level+1
            #     for ball_rep in self.ct_dict[level].keys():
            #         if ball_rep[1] != p_ball_rep[1]:
            #             dis = self.distance(p.data, ball_rep[0].data)
            #             all_neg.append((ball_rep[0], dis))
            l1, l2 = self.get_negatives(p)
            for k in l1.keys():
                o_neg.extend(l1[k])
            for k in l2.keys():
                o_neg.extend(l2[k])
            #o_neg has (node, node_id)
            if(len(o_neg)==0):
                # print("******************** WTH *************************")
                x, y = self.get_hard_negatives(p)
                o_neg.extend(y)
                # if(len(o_neg)):
                    # print("now it's not empty")
            
            #all_neg.sort(key = lambda x : x[1])
            for i in range(len(o_neg)):
                dis = self.distance(p.data, o_neg[i][0].data)
                temp.append((o_neg[i][0], dis))
            temp.sort(reverse=True, key = lambda x : x[1])
            for i in range(len(temp)):
                n_neg.append((temp[i][0], temp[i][0].node_id))
            temp.clear()

        else:
            for i in range(len(self.ct_dict[level][(all_neg[0][0], all_neg[0][0].node_id)])):
                dis = self.distance(p.data, self.ct_dict[level][(all_neg[0][0], all_neg[0][0].node_id)][i][0].data)
                temp.append((self.ct_dict[level][(all_neg[0][0], all_neg[0][0].node_id)][i][0], dis))
            temp.sort(reverse = True, key = lambda x : x[1])
            for i in range(len(temp)):
                hard_neg.append((temp[i][0], temp[i][0].node_id))
            all_neg.pop(0)
            n_neg.extend(hard_neg) 
            if (len(all_neg)==0): #this means that there was only one other ball rep at that level which got popped
                l1, l2 = self.get_negatives(p)
                for k in l1.keys():
                    all_neg.extend(l1[k])
                for k in l2.keys():
                    all_neg.extend(l2[k])
                #now all_neg has (node, node_id) pairs
                for i in range(len(all_neg)):
                    dis = self.distance(p.data, all_neg[i][0].data)
                    temp.append((all_neg[i][0], dis))
                temp.sort(reverse=True, key = lambda x : x[1])
                for i in range(len(temp)):
                    n_neg.append((temp[i][0], temp[i][0].node_id))
            else :
                for i in range(len(all_neg)):
                    # dis = self.distance(p.data, self.ct_dict[level][(all_neg[0][0], all_neg[0][0].node_id)][i][0].data)
                    n_neg.extend(self.ct_dict[level][(all_neg[i][0], all_neg[i][0].node_id )])
        #returns pos(a tuple), hard_neg(list of tuples), n_neg(list of tuples)
        #if(len(n_neg)):
            #print("nn isnt going empty from here")      
        return pos, hard_neg, n_neg
    

    #
    # Overview: get the children of cover set Qi at level i and the
    # distances of them with point p
    #
    # Input: point p to compare the distance with Qi's children, and
    # Qi_p_ds the distances of all points in Qi with p
    #
    # Output: the children of Qi and the distances of them with point
    # p
    #
    def _getChildrenDist_(self, p, Qi_p_ds, i):
        Q = sum([n.getOnlyChildren(i) for n, _ in Qi_p_ds], [])
        if 'Parallel' in dir() and self.jobs > 1 and len(Q) >= self.min_len_parallel:
            df = self.distance
            ds = Parallel(n_jobs = self.jobs)(delayed(df)(p, q.data) for q in Q)
            Q_p_ds = list(zip(Q, ds))
        else:
            Q_p_ds = [(q, self.distance(p, q.data)) for q in Q]
        
        return Qi_p_ds + Q_p_ds

    #
    # Overview: get a list of pairs <point, distance> with the k-min distances
    #
    # Input: Input cover set Q, distances of all nodes of Q to some point
    # Output: list of pairs 
    #
    def _kmin_p_ds_(self, k, Q_p_ds):
        return nsmallest(k, Q_p_ds, lambda x: x[1])

    # return the minimum distance of Q_p_ds
    def _min_ds_(self, Q_p_ds):
        return self._kmin_p_ds_(1, Q_p_ds)[0][1]

    # format the final result. If without_distance is True then it
    # returns only a list of data points, other it return a list of
    # pairs <point.data, distance>
    def _result_(self, res, without_distance):
        if without_distance:
            return [(p.data, p.node_id) for p, _ in res]
        else:
            return [(p.data, d) for p, d in res]
    
    #
    # Overview: write to a file the dot representation
    #
    # Input: None
    # Output: 
    #
    def writeDotty(self, outputFile):
        outputFile.write("digraph {\n")
        self.writeDotty_rec(outputFile, [self.root], self.maxlevel)
        outputFile.write("}")


    #
    # Overview:recursively build printHash (helper function for writeDotty)
    #
    # Input: C, i is the level
    #
    def writeDotty_rec(self, outputFile, C, i):
        if(i == self.minlevel):
            return

        children = []
        for p in C:
            childs = p.getChildren(i)

            for q in childs:
                outputFile.write("\"lev:" +str(i) + " "
                                 + str(p.data) + "\"->\"lev:"
                                 + str(i-1) + " "
                                 + str(q.data) + "\"\n")

            children.extend(childs)
        
        self.writeDotty_rec(outputFile, children, i-1)

    def __str__(self):
        output = io.StringIO()
        self.writeDotty(output)
        return output.getvalue()


    # check if the tree satisfies all invariants
    def check_invariants(self):
        return self.check_nesting() and             self.check_covering_tree() and             self.check_seperation()
    # check if my_invariant is satisfied:
    # C_i denotes the set of nodes at level i
    # for all i, my_invariant(C_i, C_{i-1})
    def check_my_invariant(self, my_invariant):
        C = [self.root]
        for i in reversed(range(self.minlevel, self.maxlevel + 1)):        
            C_next = sum([p.getChildren(i) for p in C], [])
            if not my_invariant(C, C_next, i):
                print("At level", i, "the invariant", my_invariant, "is false")
                return False
            C = C_next
        return True
        
    
    # check if the invariant nesting is satisfied:
    # C_i is a subset of C_{i-1}
    def nesting(self, C, C_next, _):
        return set(C) <= set(C_next)

    def check_nesting(self):
        return self.check_my_invariant(self.nesting)
        
    
    # check if the invariant covering tree is satisfied
    # for all p in C_{i-1} there exists a q in C_i so that
    # d(p, q) <= base^i and exactly one such q is a parent of p

    def covering_tree(self, C, C_next, i):
        return all(unique(self.distance(p.data, q.data) <= self.base**i
                          and p in q.getChildren(i)
                          for q in C)
                   for p in C_next)

    def check_covering_tree(self):
        return self.check_my_invariant(self.covering_tree)

    # check if the invariant seperation is satisfied
    # for all p, q in C_i, d(p, q) > base^i
    def seperation(self, C, _, i):
        return all(self.distance(p.data, q.data) > self.base**i
                   for p, q in product(C, C) if p != q)

    def check_seperation(self):
        return self.check_my_invariant(self.seperation)


# %%
import random
from numpy import subtract, dot, sqrt
from scipy.spatial import distance as dist
from random import random, seed
import time
import pickle as pickle

def distance(p, q):
    # print "distance"
    # print "p =", p
    # print "q =", q
    return dist.euclidean(p ,q)

def test_covertree(pts,b):
    

    total_tests = 0
    passed_tests = 0
    
    n_points = len(pts)
    m_metric = pts[0].shape[-1]

    k = 2

    dis_mat = np.zeros((n_points, n_points), dtype=float)
    for i in range(n_points):
        for j in range(n_points):
            if(i != j):
                dis_mat[i][j] = distance(pts[i], pts[j])
    
    max_dis = np.amax(dis_mat)
    print("Max dis : ", max_dis)
    min_dis = np.amin(dis_mat)
    print("Min dis : ", min_dis)
    max_l = 0
    min_l = 0
    for i in range(-4,10):
        if 2**i > max_dis:
            max_l = i
            break
    # print("max level came out to be:", max_l)
    


    gt = time.time

    # print("Build cover tree of", n_points, "in ", m_metric ," space!\n")
    
    t = gt()
    ct = CoverTree(distance, max_l)
    for p in range(len(pts)):
        ct.insert(pts[p], b[p])
    b_t = gt() - t
    # print("Building time:", b_t, "seconds")

    # #check that it is a valid cover tree
    # print("==== Check that all cover tree invariants are satisfied ====")
    if ct.check_invariants():
        # print("OK!")
        passed_tests += 1
    else:
        print("NOT OK!")
    total_tests += 1


    #see levels in  cover tree
    print("maxlevel of cover tree : ", ct.maxlevel)
    print("minlevel of cover tree : ", ct.minlevel)

    #make ball_rep of the cover tree
    ct_dict, root_kids, n , maxl, minl = ct.make_ball_rep()

    #level dic with pos and neg
    #index = actual node index, tuple[1] = node_id of pos/negative
    l_dict = {}
    # ct.minlevel+1: {'p':[], 'hn' :[], 'nn':[]}, ct.minlevel+2 : {'p':[], 'hn':[], 'nn':[]}, ct.minlevel+3 : {'p':[], 'hn' :[], 'nn':[]}
    for lev in range(minl+1, minl+4):
        l_dict[lev]={'p':[], 'hn' :[], 'nn':[]}
    

    for l in l_dict.keys():
        for point in range(len(pts)):
            p = Node(pts[point], b[point])
            pos, h_n, n_n = ct.get_pos_neg(p, l)
            l_dict[l]['p'].append(pos)
            l_dict[l]['hn'].append(h_n)
            l_dict[l]['nn'].append(n_n)

    return l_dict


# %%



# %%
def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()



def download(dataset):
    print("in download")
    basedir = os.path.dirname(os.path.abspath(''))
    print("basedir in download", basedir)
    datadir = os.path.join(basedir, 'jaynee', 'data', dataset)
    print("datadir in download", datadir)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
        url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{0}.zip'.format(dataset)
        zipfile = os.path.basename(url)
        os.system('wget {0}; unzip {1}'.format(url, zipfile))
        os.system('mv {0}/* {1}'.format(dataset, datadir))
        os.system('rm -r {0}'.format(dataset))
        os.system('rm {0}'.format(zipfile))



def process(dataset):
    print("in process")
    src = os.path.join(os.path.dirname(os.path.abspath('')), 'jaynee', 'data')
    prefix = os.path.join(src, dataset, dataset)
    print("src in process", src)
    print("prefix in process", prefix)

    graph_node_dict = {}
    #graph_node_dict-> key=node_ids , val=graph_id that node belongs to
    with open('{0}_graph_indicator.txt'.format(prefix), 'r') as f:
        #graph_indicator stores graph_id of node with node_id=i , in ith line
        for idx, line in enumerate(f):
            graph_node_dict[idx + 1] = int(line.strip('\n'))
    max_nodes = Counter(graph_node_dict.values()).most_common(1)[0][1]
    #max no of nodes that occur across all graphs in this dataset

    node_labels = []
    #list of node label for node with node_id=i in ith line
    if os.path.exists('{0}_node_labels.txt'.format(prefix)):
        with open('{0}_node_labels.txt'.format(prefix), 'r') as f:
            # ith line stores label of node with node_id i
            for line in f:
                node_labels += [int(line.strip('\n')) - 1]
            num_unique_node_labels = max(node_labels) + 1
    else:
        print('No node labels')

    node_attrs = []
    if os.path.exists('{0}_node_attributes.txt'.format(prefix)):
        with open('{0}_node_attributes.txt'.format(prefix), 'r') as f:
            for line in f:
                node_attrs.append(
                    np.array([float(attr) for attr in re.split("[,\s]+", line.strip("\s\n")) if attr], dtype=np.float)
                )
    else:
        print('No node attributes')

    graph_labels = []
    unique_labels = set()
    with open('{0}_graph_labels.txt'.format(prefix), 'r') as f:
        #ith line strores label of graph with graph_id=i
        for line in f:
            val = int(line.strip('\n'))
            if val not in unique_labels:
                unique_labels.add(val)
            graph_labels.append(val)
            #graph_labels will store N no of labels (where N is the no of graphs)
    label_idx_dict = {val: idx for idx, val in enumerate(unique_labels)}
    #label_idx_dict -> key=label itself , val =index starting from 0

    graph_labels = np.array([label_idx_dict[l] for l in graph_labels])
    #graph labels converted ro 1D array storing the index of the label for each of the N graphs


    adj_list = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    #adj_list ->dict with key = graph_id, val = list of all edges in the graph
    index_graph = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    #index_graph -> key = graph_id, val = list of all nodes in the graph
    with open('{0}_A.txt'.format(prefix), 'r') as f:
        for line in f:
            u, v = tuple(map(int, line.strip('\n').split(',')))
            adj_list[graph_node_dict[u]].append((u, v))
            index_graph[graph_node_dict[u]] += [u, v]

    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]
    #now index_graph dict has -> key=graph_id , val=list(nodes in the graph occuring only once in the list)

    graphs = []
    #list of nx_graphs
    for idx in range(1, 1 + len(adj_list)):
        graph = nx.from_edgelist(adj_list[idx])
        #creating a graph from its list of edges
        if max_nodes is not None and graph.number_of_nodes() > max_nodes:
            continue

        graph.graph['label'] = graph_labels[idx - 1]
        #assigning the label to the graph
        for u in graph.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                graph.nodes[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                graph.nodes[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            graph.graph['feat_dim'] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        #dictionary with mapping-> key= node_index starting from 0 , val = node_id
        for node_idx, node in enumerate(graph.nodes()):
            mapping[node] = node_idx

        graphs.append(nx.relabel_nodes(graph, mapping))

    if 'feat_dim' in graphs[0].graph:
        pass
    else:
        max_deg = max([max(dict(graph.degree).values()) for graph in graphs])
        #max degree encountered across all nodes over all graphs
        for graph in graphs:
            for u in graph.nodes(data=True):
                f = np.zeros(max_deg + 1)
                f[graph.degree[u[0]]] = 1.0
                if 'label' in u[1]:
                    f = np.concatenate((np.array(u[1]['label'], dtype=np.float), f))
                graph.nodes[u[0]]['feat'] = f
                #so the node features in every graph have incorporated the deg of the node & node label(if there is any)
    return graphs
  






def load(dataset):
    basedir = os.path.dirname(os.path.abspath(''))
    print("basedir inside load", basedir)
    datadir = os.path.join(basedir, 'jaynee', 'data', dataset)
    # datadir = '/home/jaynee/data/'+dataset
    print("datadir inside load", datadir)



    if not os.path.exists(datadir):
        download(dataset)
        graphs = process(dataset)
        feat, adj, labels = [], [], []

        for idx, graph in enumerate(graphs):
            adj.append(nx.to_numpy_array(graph))
            labels.append(graph.graph['label'])
            feat.append(np.array(list(nx.get_node_attributes(graph, 'feat').values())))

        adj, feat, labels = np.array(adj), np.array(feat), np.array(labels)
        #adj -> np.array of adj matrices of graphs
        #feat -> np.array of array of features of each graph
        #labels -> np.array of label of each graph

        np.save(f'{datadir}/adj.npy', adj)
        print(datadir)
        print("\n")
        np.save(f'{datadir}/feat.npy', feat)
        np.save(f'{datadir}/labels.npy', labels)

    else:
        adj = np.load(f'{datadir}/adj.npy', allow_pickle=True)
        feat = np.load(f'{datadir}/feat.npy', allow_pickle=True)
        labels = np.load(f'{datadir}/labels.npy', allow_pickle=True)

    max_nodes = max([a.shape[0] for a in adj])
    feat_dim = feat[0].shape[-1]

    num_nodes = []
    #list of no of nodes in each graph

    for idx in range(adj.shape[0]):

        num_nodes.append(adj[idx].shape[-1])

        adj[idx] = normalize_adj(adj[idx]).todense()

        adj[idx] = np.hstack(
            (np.vstack((adj[idx], np.zeros((max_nodes - adj[idx].shape[0], adj[idx].shape[0])))),
             np.zeros((max_nodes, max_nodes - adj[idx].shape[1]))))

        feat[idx] = np.vstack((feat[idx], np.zeros((max_nodes - feat[idx].shape[0], feat_dim))))

    adj = np.array(adj.tolist()).reshape(-1, max_nodes, max_nodes)
    feat = np.array(feat.tolist()).reshape(-1, max_nodes, feat_dim)

    #adj,feat and labels are column vectors (1 column vector with no of rows = no of graphs)
    return adj, feat, labels, num_nodes



def graph_embed(feat):
    g_feat = []
    feat_dim = feat[0].shape[-1]
    print("feat dim of graph embeddings ", feat_dim)
    # for i in range(feat.shape[0]):
    #     for j in range(feat.shape[1]):
    #         # for k in range(feat_dim):
    #         #     if feat[i][j][k] not in freq_dic.keys():
    #         #         freq_dic[feat[i][j][k]] = 1
    #         #     else :
    #         #         freq_dic[feat[i][j][k]] += 1
    #         for k in range(feat_dim):
    #             weight = np.random.rand()
    #             feat[i][j][k] = feat[i][j][k] * weight


    for i in range(feat.shape[0]):
        g_feat.append((np.average(feat[i],0)))
    ep = 0.001
    for i in range(len(g_feat)-1):
        for j in range(i+1,len(g_feat)):
            dis = distance(g_feat[i],g_feat[j])
            if dis==0.0:
                g_feat[j][0] += ep
                
        ep += 0.001

    #g_feat = np.array(g_feat).reshape(-1, 1, feat_dim)
    #print("shape of graph_embed mat ",g_feat.shape)
    return g_feat
    

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# %%
class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).cuda())
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).cuda())

    def forward(self, feat, adj, mask):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g
class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)
class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp1 = MLP(1 * n_h, n_h)
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.gnn2 = GCN(n_in, n_h, num_layers)

    def forward(self, adj, diff, feat, mask):
        lv1, gv1 = self.gnn1(feat, adj, mask)
        lv2, gv2 = self.gnn2(feat, diff, mask)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2
    # def get_neg_emb(self, diff_n, feat_n, mask):
    #     lv3, gv3 = self.gnn2(feat_n, diff_n, mask)

    #     lv3= self.mlp1(lv3)

    #     gv3=self.mlp2(gv3)

    #     return lv3, gv3

    def embed(self, feat, adj, diff, mask):
        __, gv1, __, gv2 = self.forward(adj, diff, feat, mask)
        return (gv1 + gv2).detach()


# %%
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure, mask):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    
    return E_neg - E_pos



def loss_function(q, k, que, gpu):
    τ = 0.95
    # N is the batch size
    N = q.shape[0]
    
    # C is the dimensionality of the representations
    C = q.shape[-1]
    #nn = que.shape[1]
    # bmm stands for batch matrix multiplication
    # If mat1 is a b×n×m tensor, mat2 is a b×m×p tensor, 
    # then output will be a b×n×p tensor. 
    # pos = torch.exp(torch.div(torch.bmm(q.view(N,1,C), k.view(N,C,1)).view(N, 1),τ))
    #que is a queue of batch_no of stacks, with each stack having k no of negatives
    l1 =[]
    l2 =[]
    for idx ,x in enumerate(q):
        l1.append(x)

    for idx, y in enumerate(k):
        l2.append(y)

    run = list(zip(l1, l2))
    # print("run len ", len(run))
    # print("run len[0] ", len(run[0][0]))
    # count = 0
    # performs matrix multiplication between query and queue tensors
    # loss_sum1 = torch.zeros(1).to(device=gpu)
    # loss_sum2 = torch.zeros(1).to(device=gpu)
    loss_sum = torch.zeros(1).to(device=gpu)
    
    for i in range(len(run)):
        # loss_sum1 = torch.zeros(1).to(device=gpu)
        # loss_sum2 = torch.zeros(1).to(device=gpu)
        sum1 = torch.zeros(1).to(device=gpu)
        # sum2 = torch.zeros(1).to(device=gpu)
        # sum2 = torch.zeros(1).to(device=gpu)
        a = run[i][0].view(1, C)
        a_n = torch.norm(a).detach()
        a = a.div(a_n.expand_as(a))
        b = run[i][1].view(C, 1)
        b_n = torch.norm(b).detach()
        b = b.div(b_n.expand_as(b))
        pos= torch.mm(a, b)
        pos=torch.div(pos, τ)
        pos=torch.exp(pos)
        # pos1=pos
        for j,x in enumerate(que[i]):
            z = x.view(C,1)
            z_n = torch.norm(z).detach()
            z = z.div(z_n.expand_as(z))
            mul1 = torch.div(torch.mm(a, z), τ)
            # mul2 = torch.div(torch.mm(b.view(1,C), z), τ)
            mul1 = torch.exp(mul1)
            # mul2 = torch.exp(mul2)
            # print("\n neg ", mul)
            sum1 =  sum1.add(mul1)
            # sum2 = sum2 + mul2

        denominator1 = pos.add(sum1)
        # denominator2 = pos1 + sum2
        loss1 = -torch.log(torch.div(pos, denominator1))
        # loss2 = -torch.log(torch.div(pos1, denominator2))
        # print("loss at batch ", loss)
        # loss_sum1 = loss_sum1 + loss1
        # loss_sum2 = loss_sum2 + loss2
        # loss_sum = loss_sum + torch.div((loss_sum1 + loss_sum2),2)
        loss_sum = loss_sum.add(loss1)
        
        # count += 1
    # sum is over positive as well as negative samples
    #denominator = neg + pos
    
    print("\n loss final ", torch.div(loss_sum, N).view([]) )
    #return torch.mean(-torch.log(torch.div(pos,denominator)))
    return torch.div(loss_sum, N).view([])
    # 




# %%
#to test hierrarchical ct pos & neg
def train_our(dataset, gpu, num_layer=4, epoch=40, batch=64):
    nb_epochs = epoch
    batch_size = batch
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 512

    #adj, diff, feat, labels, num_nodes = load(dataset)
    #load everything
    adj, feat, labels, num_nodes = load(dataset)
    
    similar = adj
    m_n = adj[0].shape[-1]
    #get graph embeddings to make a cover tree in metric space
    g_feat = graph_embed(feat)
    g_feat = np.array(g_feat).reshape(-1, 1 , feat.shape[-1])

    #just to get pos matrices
    dis_mat = np.zeros((g_feat.shape[0], g_feat.shape[0]), dtype=float)
    for i in range(g_feat.shape[0]):
        for j in range(g_feat.shape[0]):
            if(i != j):
                dis_mat[i][j] = distance(g_feat[i], g_feat[j])
    
    max_dis = np.amax(dis_mat)
    # print("Max dis : ", max_dis)
    min_dis = np.amin(dis_mat)
    # print("Min dis : ", min_dis)
    max_l = 0
    min_l = 0
    for i in range(-4,7):
        if 2**i > max_dis:
            max_l = i
            break
    # print("max level came out to be:", max_l)
    pct = CoverTree(distance, max_l)
    #make ct for entire dataset to get pos matrices
    for pt in range(len(g_feat)):
        #print(pt)
        pct.insert(g_feat[pt], pt)
    # print("g_feat : ", len(g_feat))

    xpos=[]
    for i in range(len(g_feat)):
        query=g_feat[i]
        results = pct.knn(2, query, True)
        #print("2 nearest neighbour of \n",query,"\nis \n", results)
        xpos.append(results[1][1])
        results.clear()
    diff = []
    for i in range(len(xpos)):
        diff.append(similar[xpos[i]-1])
    
    diff = np.array(diff).reshape(-1, adj.shape[-1], adj.shape[-1])

    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]

    model = Model(ft_size, hid_units, num_layer)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model.cuda()

    feat = torch.FloatTensor(feat).cuda()
    adj = torch.FloatTensor(adj).cuda()
    diff = torch.FloatTensor(diff).cuda()
    labels = torch.LongTensor(labels).cuda()

    cnt_wait = 0
    best = 1e9

    total = 0
    no_pos = 0
    no_neg = 0

    itr = (adj.shape[0] // batch_size) + 1
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        train_idx = np.arange(adj.shape[0])
        np.random.shuffle(train_idx)
        
        for idx in range(0, len(train_idx), batch_size):
            model.train()
            optimiser.zero_grad()

            batch = train_idx[idx: idx + batch_size]
            mask = num_nodes[idx: idx + batch_size]
            bt = batch
            

            #get positives, negatives and further embeddings from the cover tree
            ct1 = test_covertree(g_feat[bt],bt)

           
            
           
            #to get losses for three levels
            lo = []
            for lev in ct1.keys():
                # print("ct1 for level : ",lev)
                # print("len of xpos at this lev :", len(ct1[lev]['p']))
                # print("len of hn at this lev :", len(ct1[lev]['hn']))
                # print("len of nn at this lev :", len(ct1[lev]['nn']))

                # pos = list()
                # print("batch lenth : ",len(bt))
                # for i in range(len(bt)):
                #     pos.append(similar[ct1[lev]['p'][i][1]])
                # print("top :", type(pos))
                # temp = np.array(pos)
                # temp = temp.reshape(-1, m_n, m_n)

                # diff = torch.FloatTensor(temp).cuda()
                diff = diff.cpu().numpy()
                adj = adj.cpu().numpy()

                total += 1

                for x in range(len(bt)):
                    diff[bt[x]]=adj[ct1[lev]['p'][x][1]]
                    if (labels[bt[x]] == labels[ct1[lev]['p'][x][1]]):
                        no_pos += 1
                    for i in range(len(ct1[lev]['nn'][x])):
                        if (labels[bt[x]] != labels[ct1[lev]['nn'][x][i][1]]):
                            no_neg += 1
                        while(i<=10):
                            no_neg += 1
                            i += 1
                diff = torch.FloatTensor(diff).cuda()
                adj = torch.FloatTensor(adj).cuda()

                lv1, gv1, lv2, gv2 = model(adj[bt], diff[bt], feat[bt], mask)

                # print("gv1 size ", gv1.size())
                #lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
                #lv2 = lv2.view(batch.shape[0] * max_nodes, -1)

                que = []
                for b_id in range(len(bt)):
                    temp_list = []
                    for i in range(10):
                        for j in range(len(bt)):
                            if len(ct1[lev]['nn'][b_id])>0:
                                #print("it is >0")
                                if len(ct1[lev]['nn'][b_id])>i :
                                    if ct1[lev]['nn'][b_id][i][1]== bt[j]:
                                        x=gv2[j]
                                        temp_list.append(x)
                                        break
                                else:
                                    temp_list.append(temp_list[0])
                    # if(len(ct1[lev]['nn'][b_id])==0):
                        # print("nn is still empty -_-")
                    # if(len(temp_list)<5):
                        # print("b_id ", bt[b_id]," with batch index ",b_id," has ",len(temp_list)," nn!")
                    # x = torch.stack(temp_list)
                    # mean = x.mean(axis = 0)
                    # std = x.std(axis = 0)
                    # # l = list()
                    # # l.append(mean)
                    # # l.append(std)
                    # # ip = torch.stack(l)
                    # # newt = torch.sum(ip, dim=0)
                    # newt = mean + std
                    # amt = -0.1
                    # for t in range(5):
                    #     newt = torch.add(newt, amt)
                    #     amt = amt -0.1
                    #     temp_list.append(newt)

                    que.append(torch.stack(temp_list))
                    temp_list.clear()
                

                             
                
                #lv3, gv3, lv4, gv4 = model(adj[batch], adj[n_batch], feat[n_batch], mask)
            

                #loss1 = local_global_loss_(lv1, gv2, 'JSD', mask)
                #loss2 = local_global_loss_(lv2, gv1, batch, 'JSD', mask)
                #loss3 = global_global_loss_(gv1, gv2, 'JSD')
                lo.append(loss_function(gv1, gv2, que, gpu))
                
                
            batch = torch.LongTensor(np.repeat(np.arange(bt.shape[0]), max_nodes)).cuda()
            #loss = loss1 + loss2 #+ loss3
            loss = lo[0]
            for i in range(1,len(lo)):
                loss += lo[i]

            epoch_loss += loss
            loss.backward()
            optimiser.step()

        epoch_loss /= itr

        # print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, epoch_loss))

        if epoch_loss < best:
            best = epoch_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), f'{dataset}-{gpu}.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            break

    model.load_state_dict(torch.load(f'{dataset}-{gpu}.pkl'))

    features = feat.cuda()
    adj = adj.cuda()
    diff = diff.cuda()
    labels = labels.cuda()

    print("shape of feats :", features.shape)
    print("shape of adj :", adj.shape)
    print("shape of diff :", diff.shape)

    embeds = model.embed(features, adj, diff, num_nodes)

    x = embeds.cpu().numpy()
    y = labels.cpu().numpy()

    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    for train_index, test_index in kf.split(x, y):

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
    # print("##############################################################################")
    print("##############################################################################")
    print(np.mean(accuracies), np.std(accuracies))
    print("##############################################################################")
    print("Percentage of positives : ")
    print((no_pos/total) * 100 )
    print("##############################################################################")
    print("Percentage of negatives : ")
    print((no_neg/(total*10)) * 100 )
    print("##############################################################################")
    return np.mean(accuracies), np.std(accuracies)


# %%
#main hierarchical test 1
if __name__ == '__main__':
    #training 
    import warnings
    warnings.filterwarnings("ignore")
    gpu = 0
    torch.cuda.set_device(gpu)
    layers = [4]
    batch = [128]
    epoch = [20]
    ds = ['PTC_MR']
    
    #'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'
    seeds = [132]
    #,123,312,231,321
    #for d in ds:
    accuracies = []
    std = []
    #best {'l':0,'b':0,'e':0,'s':0}
    #print("training for MUTAG dataset ")
    for d in ds:
        print(f'####################{d}####################')
        for l in layers:
            for b in batch:
                for e in epoch:
                    for i in range(len(seeds)):
                        seed = seeds[i]
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print('################################################')
                        print(f' Layer:{l}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        a, di = train_our(d, gpu, l, e, b)
                        accuracies.append(a)
                        std.append(di)
                        print('################################################')
        print("\n Final Accuracy and STD : \n")
        a_max = 0
        std_max = 0
        for i in range(len(accuracies)):
            if accuracies[i]>a_max:
                a_max = accuracies[i]
                std_max = std[i]
        print("accuracy : ", a_max  ," and std : ", std_max)
