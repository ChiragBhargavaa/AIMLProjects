graph = {
'M': ['C', 'H'],
'C': ['M', 'D', 'E'],
'H': ['M', 'S'],
'D': ['C', 'L'],
'E': ['C', 'A'],
'S': ['H', 'E'],
'L': ['D', 'C'],
'A': ['E']
}

def dfs(graph , start , visited , order):
    visited.add(start)
    order.append(start)
    for neighbour in graph[start]:
        if neighbour not in visited:
            dfs(graph , neighbour , visited , order)

order =[]
visited = set()
dfs(graph , 'M',visited , order)
print(order)
            