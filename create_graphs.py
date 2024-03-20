import numpy as np
import networkx as nx

class Graph:
    def __init__(self, vertix=4) -> None:
        self.vertix = vertix
        self.array = np.random.randint(2, size=(self.vertix, self.vertix))
    
    def get_matrix(self):
        return self.array
    
    def get_graph(self):
        return nx.from_numpy_array(self.array)
    
    def get_chrome_number(self):
        return len(set(nx.coloring.greedy_color(self.get_graph()).values()))

    
if __name__ == "__main__":
    
    graph_list = []
    for _ in range(3):
        graph_list.append(Graph(5))

    for i in graph_list:
        print(i.get_graph())