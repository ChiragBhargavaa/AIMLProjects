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

def dfs(graph , start):
    order = []
    visited = set()
    queue = []

    queue.append(start)
    visited.add(start)

    while queue:
        current = queue.pop(0)
        order.append(current)
        for neighbour in graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(neighbour)
    return order

print(dfs(graph, 'M'))