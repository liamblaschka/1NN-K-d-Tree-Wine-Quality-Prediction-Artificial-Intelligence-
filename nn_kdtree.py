import argparse
import pandas as pd
import numpy as np

class Node:
    def __init__(self, d, value, point):
        self.d = d
        self.value = value
        self.point = point
        self.left = None
        self.right = None

def BuildKdTree(P, D, first=False):
    if len(P) == 0:
        return None
    
    M = len(P[0]) - 1
    

    if len(P) == 1:
        d = D % M
        
        return Node(d, P[0][d], P[0])
    else:
        d = D % M
        sorted_P = P[P[:, d].argsort()]
        median = sorted_P[(len(P) // 2), d]
        
        left_P = sorted_P[sorted_P[:, d] < median] # < median
        right_P = sorted_P[sorted_P[:, d] >= median] # >= median
        
        point = right_P[0] # point at the median
        right_P = right_P[1:] # exclude this point from the right subtree
        value = point[d]
        node = Node(d, value, point)
        
        # Print number of points in left and right trees after first split of k-d tree
        if first:
            print(f"{'.' * D}l{len(left_P)}")
            print(f"{'.' * D}r{len(right_P)}")
        
        node.left = BuildKdTree(left_P, D + 1) # < median
        node.right = BuildKdTree(right_P, D + 1) # >= median
        
        
        return node


def euclidean_distance(p_x, p_y):
    distance = 0
    for i in range(len(p_x)):
        distance += (p_x[i] - p_y[i]) ** 2
    
    distance **= 0.5
    return distance
    
def nn_search(node, point, neighbour=None):
    if node is None:
        return neighbour
    
    if (neighbour is None) or (euclidean_distance(point, node.point) < euclidean_distance(point, neighbour)):
        neighbour = node.point
    
    next = None
    if point[node.d] >= node.value:
        child = node.right
        other_child = node.left
    else:
        child = node.left
        other_child = node.right
        
    neighbour = nn_search(child, point, neighbour)
    if abs(point[node.d] - node.value) < euclidean_distance(point, neighbour):
        neighbour = nn_search(other_child, point, neighbour)
            
    return neighbour


        
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path')
    parser.add_argument('test_path')
    parser.add_argument('dimension')
    args = parser.parse_args()
    train_path = args.train_path
    test_path = args.test_path
    dimension = int(args.dimension)
    
    
    train = pd.read_csv(train_path, delim_whitespace=True)
    test = pd.read_csv(test_path, delim_whitespace=True)
    
    P = train.values
    
    kd_tree = BuildKdTree(P, dimension, True)
    
    
    quality_predictions = []
    for point in test.itertuples(index=False):
        neighbour = nn_search(kd_tree, point)
        quality = int(neighbour[-1])
        quality_predictions.append(quality)
    
    for prediction in quality_predictions:
        print(prediction)
    


    
