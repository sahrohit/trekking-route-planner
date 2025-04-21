# ——————————————————————————————————————————————————————————————
# 1) load_nodes: read node importance from CSV, allow commas in name
# ——————————————————————————————————————————————————————————————
def load_nodes(path):
    node_importance = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines[1:]:
        # split on the last 4 commas, so the 'name' can contain commas
        parts = line.rsplit(",", 4)
        if len(parts) != 5:
            raise ValueError(f"Malformed nodes line: {line!r}")
        name, lat, lon, alt, imp = parts
        node_importance[name] = float(imp)
    return node_importance


# ——————————————————————————————————————————————————————————————
# 2) load_edges: read edges (distance used as time_cost), allow commas in node names
# ——————————————————————————————————————————————————————————————
def load_edges(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for line in lines[1:]:
        # split on the last 3 commas, so source/target can contain commas
        parts = line.rsplit(",", 3)
        if len(parts) != 4:
            raise ValueError(f"Malformed edges line: {line!r}")
        src, tgt, dist, alt_diff = parts
        edges.append((src, tgt, float(dist)))
    return edges


# ——————————————————————————————————————————————————————————————
# (rest of your code is unchanged)
# ——————————————————————————————————————————————————————————————


def build_graph(node_importance, edges):
    graph = {}
    for src, tgt, dist in edges:
        graph.setdefault(src, []).append((tgt, dist, node_importance.get(tgt, 0.0)))
        graph.setdefault(tgt, []).append((src, dist, node_importance.get(src, 0.0)))
    return graph


def constrained_max_importance_path(graph, source, dest, max_time):
    best = {"importance": float("-inf"), "path": []}

    def dfs(node, time_left, visited, path, imp_sum):
        if node == dest:
            if imp_sum > best["importance"]:
                best["importance"], best["path"] = imp_sum, path.copy()
            return
        for nbr, t_cost, imp in graph.get(node, []):
            if nbr in visited or t_cost > time_left:
                continue
            visited.add(nbr)
            path.append(nbr)
            dfs(nbr, time_left - t_cost, visited, path, imp_sum + imp)
            path.pop()
            visited.remove(nbr)

    dfs(source, max_time, {source}, [source], 0.0)
    return best["path"], best["importance"]


if __name__ == "__main__":
    NODES_CSV = "dataset/nodes.csv"
    EDGES_CSV = "dataset/edges.csv"
    SOURCE = "Jiri Hospital"
    DEST = "Jiri-02"
    MAX_TIME = 5 * 6 * 3600

    node_imp = load_nodes(NODES_CSV)
    raw_edges = load_edges(EDGES_CSV)
    graph = build_graph(node_imp, raw_edges)

    path, imp = constrained_max_importance_path(graph, SOURCE, DEST, MAX_TIME)
    print(f"Best path within {MAX_TIME}s: {' → '.join(path)}")
    print(f"Total importance: {imp:.4f}")
