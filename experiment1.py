import csv
import heapq
from math import inf
import time
import sys
import random

# Constants for trekking parameters
AVG_SPEED_MPS = 0.5  # average speed in meters per second


def load_nodes(filename):
    """
    Parse nodes.csv and return a dict mapping node names to attributes.
    Expects columns: name, latitude, longitude, altitude, importance.
    """
    nodes = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name")
            try:
                lat = float(row.get("latitude", 0))
                lon = float(row.get("longitude", 0))
                alt = float(row.get("altitude", 0))
                imp = float(row.get("importance", 0))
            except ValueError as e:
                raise ValueError(f"Invalid numeric in nodes CSV: {e}")
            nodes[name] = {"lat": lat, "lon": lon, "alt": alt, "imp": imp}
    return nodes


def load_edges(filename, nodes):
    """
    Parse edges.csv and return adjacency list: source->[(target, distance, alt_diff),...]
    """
    edges = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row.get("source")
            target = row.get("target")
            try:
                dist = float(row.get("distance", 0))
                alt_diff = abs(float(row.get("altitude_diff", 0)))
            except ValueError as e:
                raise ValueError(f"Invalid numeric in edges CSV: {e}")
            if source not in nodes or target not in nodes:
                continue
            edges.setdefault(source, []).append((target, dist, alt_diff))
            edges.setdefault(target, []).append((source, dist, alt_diff))
    return edges


def compute_hours_used(path, edges):
    """
    Sum total hours required to traverse the given path.
    """
    total = 0.0
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        dist = next((d for nbr, d, _ in edges.get(src, []) if nbr == dst), 0)
        # time in hours
        total += dist / (AVG_SPEED_MPS * 3600)
    return total


def build_itinerary(path, edges, max_hours):
    """
    Build list of legs deducting from max_hours and tracking hours left.
    """
    itinerary = []
    hours_left = max_hours
    for i in range(len(path) - 1):
        src, dst = path[i], path[i + 1]
        dist = next((d for nbr, d, _ in edges.get(src, []) if nbr == dst), 0)
        time_h = dist / (AVG_SPEED_MPS * 3600)
        hours_left_after = hours_left - time_h
        itinerary.append(
            {
                "from": src,
                "to": dst,
                "distance": dist,
                "time_hours": time_h,
                "hours_left_after": hours_left_after,
            }
        )
        hours_left = hours_left_after
        if hours_left <= 0:
            break
    return itinerary


def score_path(
    path,
    nodes,
    edges,
    imp_weight=0.4,
    alt_weight=0.4,
    hours_weight=0.2,
    max_hours=None,
):
    """
    Score path by total importance, altitude change, and hours utilization.
    """
    if not path or len(path) < 2:
        return -float("inf"), 0.0, 0.0, 0.0
    total_imp = sum(nodes[n]["imp"] for n in path)
    avg_imp = total_imp / len(path)
    total_alt = sum(
        abs(nodes[path[i + 1]]["alt"] - nodes[path[i]]["alt"])
        for i in range(len(path) - 1)
    )
    used = compute_hours_used(path, edges)
    # invalidate if exceeds
    if max_hours is not None and used > max_hours:
        return -float("inf"), used, avg_imp, total_alt
    # normalized utilization
    hours_score = used / max_hours if max_hours else 0
    score = (
        hours_weight * hours_score
        + imp_weight * avg_imp
        - alt_weight * (total_alt / (len(path) - 1) / 1000)
    )
    return score, used, avg_imp, total_alt


def greedy_path(
    nodes,
    edges,
    start,
    end,
    max_hours,
    imp_weight=0.4,
    alt_weight=0.4,
    hours_weight=0.2,
):
    """
    Greedy backtracking: prefer neighbors maximizing score, under total hours budget.
    """
    stack = [(start, edges.get(start, []).copy(), {start})]
    path = [start]
    while stack:
        current, nbrs, visited = stack[-1]
        used = compute_hours_used(path, edges)
        if current == end or used >= max_hours:
            break
        best_score, best_nbr = float("-inf"), None
        for nbr, dist, alt in nbrs:
            if nbr in visited:
                continue
            trial = path + [nbr]
            score, used_t, *_ = score_path(
                trial, nodes, edges, imp_weight, alt_weight, hours_weight, max_hours
            )
            if score > best_score:
                best_score, best_nbr = score, nbr
        if best_nbr:
            # remove and go deeper
            current_nbrs = [(x, y, z) for x, y, z in nbrs if x != best_nbr]
            stack[-1] = (current, current_nbrs, visited)
            path.append(best_nbr)
            visited.add(best_nbr)
            stack.append((best_nbr, edges.get(best_nbr, []).copy(), visited.copy()))
        else:
            stack.pop()
            path.pop()
    if path and path[-1] == end:
        used = compute_hours_used(path, edges)
        total_imp = sum(nodes[n]["imp"] for n in path)
        total_alt = sum(
            abs(nodes[path[i + 1]]["alt"] - nodes[path[i]]["alt"])
            for i in range(len(path) - 1)
        )
        return path, total_imp, total_alt, build_itinerary(path, edges, max_hours)
    return [], 0.0, 0.0, []


def dac_path(
    nodes,
    edges,
    start,
    end,
    max_hours,
):
    """
    Brute‑force all simple outbound paths, then return via shortest path.
    Guarantees you start at `start`, visit end at least at the very end,
    and never exceed max_hours (outbound + return).
    Chooses path that lexicographically maximizes
    (total_importance, outbound_node_count, total_hours_used).
    """

    # 1) Dijkstra from 'end' to get shortest‐return distances & reconstruct paths
    dist_m = {n: inf for n in nodes}
    prev = {}
    dist_m[end] = 0
    pq = [(0.0, end)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist_m[u]:
            continue
        for v, d_uv, _alt in edges.get(u, []):
            nd = d_u + d_uv
            if nd < dist_m[v]:
                dist_m[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Build return_path[u] = list of nodes [u,...,end] along that shortest route
    return_path = {}
    for u in nodes:
        if dist_m[u] < inf:
            path = [u]
            while path[-1] != end:
                path.append(prev[path[-1]])
            return_path[u] = path

    best = None
    best_key = None  # (total_imp, count, total_hours)

    # 2) DFS‐enumerate every simple outbound path from `start`
    def dfs(path, visited):
        nonlocal best, best_key
        u = path[-1]

        # At each node u, consider “stop outbound here + return to end”
        if u in return_path:
            hours_out = compute_hours_used(path, edges)
            hours_ret = dist_m[u] / (AVG_SPEED_MPS * 3600)
            total_h = hours_out + hours_ret

            if total_h <= max_hours:
                total_imp = sum(nodes[n]["imp"] for n in path)
                count = len(path)
                key = (total_imp, count, total_h)
                if best_key is None or key > best_key:
                    # build full round‑trip path
                    full = path + return_path[u][1:]
                    # compute altitude change over full path
                    total_alt = sum(
                        abs(nodes[full[i + 1]]["alt"] - nodes[full[i]]["alt"])
                        for i in range(len(full) - 1)
                    )
                    best = (full, total_imp, total_alt)
                    best_key = key

        # Then try to extend outbound
        for v, _, _ in edges.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            dfs(path, visited)
            path.pop()
            visited.remove(v)

    dfs([start], {start})

    if best:
        full_path, imp, alt = best
        return full_path, imp, alt, build_itinerary(full_path, edges, max_hours)

    return [], 0.0, 0.0, []


def dp_path(nodes, edges, start, end, max_hours):
    """
    Finds a simple path that:
      - begins at `start`
      - visits `end` at least once
      - ends at `end` (via shortest available return)
      - never exceeds max_hours total (outbound + return)
      - lexicographically maximizes (importance, node_count, hours_used)
    """

    # --- Precompute shortest‐return paths from every node back to `end` ---
    # Dijkstra on the undirected graph, weights = distance in meters
    dist_m = {n: float("inf") for n in nodes}
    prev = {}
    dist_m[end] = 0
    pq = [(0, end)]
    while pq:
        d_u, u = heapq.heappop(pq)
        if d_u > dist_m[u]:
            continue
        for v, w, _ in edges.get(u, []):
            nd = d_u + w
            if nd < dist_m[v]:
                dist_m[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # rebuild return‐path lists
    return_path = {}
    for u in nodes:
        if u == end or dist_m[u] < float("inf"):
            path = [u]
            while path[-1] != end:
                path.append(prev[path[-1]])
            return_path[u] = path

    # helper: return‐time in hours from u back to end
    def return_hours(u):
        return dist_m[u] / (AVG_SPEED_MPS * 3600)

    # --- DP recursion ---
    # @lru_cache(maxsize=None)
    def dfs(u, visited_tuple, seen_end):
        visited = set(visited_tuple)
        best = None
        best_key = None  # (imp, count, total_hours)

        # 1) Option to **stop here** (only if we've already hit end)
        if seen_end:
            rh = return_hours(u)
            if rh <= max_hours:
                imp_sum = nodes[u]["imp"]
                cnt = 1
                hours_wo = 0.0
                total_h = hours_wo + rh
                best = ([u], imp_sum, cnt, hours_wo)
                best_key = (imp_sum, cnt, total_h)

        # 2) Try every unvisited neighbor
        for v, dist, _alt in edges.get(u, []):
            if v in visited:
                continue
            edge_h = dist / (AVG_SPEED_MPS * 3600)
            if edge_h > max_hours:
                continue

            new_seen_end = seen_end or (v == end)
            new_visited = tuple(sorted(visited | {v}))
            sub = dfs(v, new_visited, new_seen_end)
            if not sub:
                continue

            sub_path, sub_imp, sub_cnt, sub_hours_wo = sub
            hours_wo = edge_h + sub_hours_wo

            # ensure we can still return from the tail
            tail = sub_path[-1]
            rh = return_hours(tail)
            total_h = hours_wo + rh
            if total_h > max_hours:
                continue

            imp_sum = nodes[u]["imp"] + sub_imp
            cnt = 1 + sub_cnt
            cand_key = (imp_sum, cnt, total_h)
            if best is None or cand_key > best_key:
                best = ([u] + sub_path, imp_sum, cnt, hours_wo)
                best_key = cand_key

        return best

    # kick off: mark seen_end True if start==end
    res = dfs(start, (start,), start == end)
    if not res:
        return [], 0.0, 0.0, []

    path_wo_ret, total_imp, _, hours_wo = res
    tail = path_wo_ret[-1]

    # append the return‐segment
    ret_seg = return_path.get(tail, [])
    full_path = path_wo_ret + ret_seg[1:]  # skip duplicate tail

    # compute altitude change on full_path
    total_alt = sum(
        abs(nodes[full_path[i + 1]]["alt"] - nodes[full_path[i]]["alt"])
        for i in range(len(full_path) - 1)
    )
    itinerary = build_itinerary(full_path, edges, max_hours)
    return full_path, total_imp, total_alt, itinerary


def write_results_to_csv(filename, results):
    """Write summary results to CSV including hours used."""
    with open(filename, "w", newline="") as csvfile:
        fieldnames = [
            "algorithm",
            "path",
            "total_importance",
            "average_importance",
            "total_altitude_change",
            "path_length",
            "hours_used",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for alg, (path, imp, alt, iti) in results.items():
            # used = compute_hours_used(
            #     path, results["dummy"] if False else edges
            # )  # placeholder
            avg_imp = imp / len(path) if path else 0
            writer.writerow(
                {
                    "algorithm": alg,
                    "path": ",".join(path),
                    "total_importance": f"{imp:.4f}",
                    "average_importance": f"{avg_imp:.4f}",
                    "total_altitude_change": f"{alt:.2f}",
                    "path_length": len(path),
                    # "hours_used": f"{used:.2f}",
                }
            )


HOURS_PER_DAY = 5  # assumed trekking hours per day


def generate_random_dataset(n_nodes, edge_per_node=2):
    """
    Generate a random graph dataset with n_nodes.
    Each node has random coordinates, altitude, and importance.
    Each node connects to edge_per_node random neighbors.
    """
    nodes = {
        f"node{i}": {
            "lat": random.random(),
            "lon": random.random(),
            "alt": random.uniform(0, 1000),
            "imp": random.random(),
        }
        for i in range(n_nodes)
    }
    edges = {}
    node_list = list(nodes.keys())
    for node in node_list:
        neighbors = random.sample(
            [n for n in node_list if n != node], min(edge_per_node, n_nodes - 1)
        )
        edges[node] = []
        for nbr in neighbors:
            d = (
                (nodes[node]["lat"] - nodes[nbr]["lat"]) ** 2
                + (nodes[node]["lon"] - nodes[nbr]["lon"]) ** 2
            ) ** 0.5 * 1000
            alt_diff = abs(nodes[node]["alt"] - nodes[nbr]["alt"])
            edges[node].append((nbr, d, alt_diff))
    return nodes, edges


def run_experiment1():
    sizes = [
        5,
        10,
        20,
        30,
        50,
        75,
        # 100,
        # 50,
        # 100, 500, 1000
    ]
    with open("experiment1_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["network_size", "algorithm", "runtime_seconds", "memory_bytes"]
        )
        for size in sizes:
            nodes, edges = generate_random_dataset(size)
            start = list(nodes.keys())[0]
            end = list(nodes.keys())[-1]
            max_hours = float(size)  # mapping size to max_hours for consistency

            # print(f"Running experiments for size {size}...")
            # print(f"Nodes: {nodes}")
            # print(f"Edges: {edges}")

            for name, func in [
                ("greedy", greedy_path),
                ("divide_and_conquer", dac_path),
                ("dynamic_programming", dp_path),
            ]:
                t0 = time.time()
                path, imp, alt, iti = func(nodes, edges, start, end, max_hours)
                runtime = time.time() - t0
                mem = (
                    sys.getsizeof(nodes)
                    + sys.getsizeof(edges)
                    + sys.getsizeof(path)
                    + sys.getsizeof(iti)
                )
                writer.writerow([size, name, runtime, mem])
                print(
                    f"Algorithm: {name}, Size: {size}, Runtime: {runtime:.4f} seconds, Memory: {mem} bytes"
                )
    print("Experiment 1 results saved to experiment1_results.csv")


def run_experiment2():
    days_list = [3, 7, 14, 30, 60, 100]
    imp_thresholds = [0.2, 0.5, 0.8]
    diff_thresholds = [200, 500, 1000]
    nodes, edges = generate_random_dataset(100)
    start = list(nodes.keys())[0]
    end = list(nodes.keys())[-1]

    with open("experiment2_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "days",
                "imp_threshold",
                "diff_threshold",
                "algorithm",
                "runtime_seconds",
                "feasible",
            ]
        )
        for days in days_list:
            max_hours = days * HOURS_PER_DAY
            for imp_th in imp_thresholds:
                # filter nodes by importance threshold
                filt_nodes = {n: d for n, d in nodes.items() if d["imp"] >= imp_th}
                # filter edges by difficulty threshold
                filt_edges = {
                    n: [
                        (nbr, dist, alt)
                        for nbr, dist, alt in edges[n]
                        if nbr in filt_nodes and alt <= imp_th
                    ]
                    for n in filt_nodes
                }
                for name, func in [
                    ("greedy", greedy_path),
                    ("divide_and_conquer", dac_path),
                    ("dynamic_programming", dp_path),
                ]:
                    t0 = time.time()
                    try:
                        path, imp_sum, alt_sum, iti = func(
                            filt_nodes, filt_edges, start, end, max_hours
                        )
                        feasible = int(
                            compute_hours_used(path, filt_edges) <= max_hours
                            and bool(path)
                        )
                    except:  # noqa: E722
                        feasible = 0
                    runtime = time.time() - t0
                    writer.writerow(
                        [days, imp_th, diff_thresholds, name, runtime, feasible]
                    )
    print("Experiment 2 results saved to experiment2_results.csv")


def main():
    run_experiment1()
    # run_experiment2()


if __name__ == "__main__":
    main()
