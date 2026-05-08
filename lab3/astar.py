import heapq

graph = {
    'C': [('A', 2), ('B', 1), ('D', 3)],
    'A': [('C', 2), ('E', 4), ('F', 6)],
    'B': [('C', 1), ('D', 2), ('H', 5)],
    'D': [('C', 3), ('B', 2), ('I', 4)],
    'E': [('A', 4), ('J', 3)],
    'F': [('A', 6), ('K', 2)],
    'H': [('B', 5), ('I', 2), ('L', 6)],
    'I': [('D', 4), ('H', 2), ('M', 3)],
    'J': [('E', 3), ('G', 4)],
    'K': [('F', 2), ('G', 3)],
    'L': [('H', 6), ('M', 2)],
    'M': [('I', 3), ('L', 2), ('G', 1)],
    'G': [('J', 4), ('K', 3), ('M', 1)]
}

h = {
    'C': 6, 'A': 7, 'B': 6, 'D': 5,
    'E': 6, 'F': 4, 'H': 5, 'I': 4,
    'J': 4, 'K': 3, 'L': 3, 'M': 1,
    'G': 0
}

def astar(start, goal):
    open_list=[]
    g={start:0}
    parent={start:None}
    heapq.heappush(open_list, (h[start],start))

    while open_list:
        f, current = heapq.heappop(open_list)
        if current == goal:
            break

        for neighbour,cost in graph[current]:
            newg= g[current]+cost 
            if neighbour not in g or newg < g[neighbour]:
                parent[neighbour]= current
                newf = newg + h[neighbour]
                g[neighbour]= newg
                heapq.heappush(open_list, (newf,neighbour))

    path=[]
    node= goal
    while node is not None:
        path.append(node)
        node=parent[node]

    path.reverse()
    print(path)


   
start='C'
goal='G'
astar(start,goal)