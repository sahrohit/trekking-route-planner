import csv


def load_nodes(filename):
    """
    Parse nodes.csv and return a dict mapping node names to their attributes.
    Expects columns: name, latitude, longitude, altitude, importance.
    """
    nodes = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                name = row["name"]
                lat = float(row["latitude"])
                lon = float(row["longitude"])
                alt = float(row["altitude"])
                imp = float(row["importance"])
            except KeyError as e:
                raise ValueError(f"Missing column in nodes CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Invalid numeric value in nodes CSV: {e}")
            nodes[name] = {"lat": lat, "lon": lon, "alt": alt, "imp": imp}
    return nodes


def load_edges(filename, nodes):
    """
    Parse edges.csv and build an undirected adjacency list.
    Expects columns: source, target, distance, altitude_diff.
    Filters out edges involving unknown nodes.
    """
    edges = {}
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                source = row["source"]
                target = row["target"]
                dist = float(row["distance"])
                alt_diff = abs(float(row["altitude_diff"]))
            except KeyError as e:
                raise ValueError(f"Missing column in edges CSV: {e}")
            except ValueError as e:
                raise ValueError(f"Invalid numeric value in edges CSV: {e}")
            if source not in nodes or target not in nodes:
                continue
            edges.setdefault(source, []).append((target, dist, alt_diff))
            edges.setdefault(target, []).append((source, dist, alt_diff))
    return edges


def greedy_path(nodes, edges, start, end, max_days, weight=1e-5):
    """
    Greedy approach: at each step, choose the neighbor that maximizes
    importance minus a penalty proportional to altitude change.
    """
    path = [start]
    total_imp = nodes[start]["imp"]
    total_alt = 0
    current = start
    days = 0
    visited = {start}

    while current != end and days < max_days:
        best = None
        best_score = float("-inf")
        for nbr, dist, alt_diff in edges.get(current, []):
            if nbr in visited:
                continue
            score = nodes[nbr]["imp"] - weight * alt_diff
            if score > best_score:
                best_score = score
                best = (nbr, alt_diff)
        if not best:
            break
        nbr, alt_diff = best
        path.append(nbr)
        total_imp += nodes[nbr]["imp"]
        total_alt += alt_diff
        visited.add(nbr)
        current = nbr
        days += 1

    return path, total_imp, total_alt


def dac_path(nodes, edges, start, end, max_days, weight=1e-5):
    """
    Divide and conquer (brute-force) recursive search without memoization.
    """

    def recurse(node, days_left, visited):
        if days_left < 0:
            return None
        if node == end:
            return ([end], nodes[end]["imp"], 0)

        best_score = float("-inf")
        best_result = None
        for nbr, dist, alt_diff in edges.get(node, []):
            if nbr in visited:
                continue
            result = recurse(nbr, days_left - 1, visited | {nbr})
            if result:
                path_rec, imp_rec, alt_rec = result
                total_imp = nodes[node]["imp"] + imp_rec
                total_alt = alt_diff + alt_rec
                score = total_imp - weight * total_alt
                if score > best_score:
                    best_score = score
                    best_result = ([node] + path_rec, total_imp, total_alt)
        return best_result

    result = recurse(start, max_days, {start})
    return result if result else ([], 0, 0)


def dp_path(nodes, edges, start, end, max_days, weight=1e-5):
    """
    Dynamic programming with memoization over (node, days_left).
    """
    memo = {}

    def recurse(node, days_left):
        key = (node, days_left)
        if key in memo:
            return memo[key]

        if days_left == 0:
            if node == end:
                memo[key] = ([end], nodes[end]["imp"], 0)
            else:
                memo[key] = None
            return memo[key]

        best_score = float("-inf")
        best_result = None
        for nbr, dist, alt_diff in edges.get(node, []):
            result = recurse(nbr, days_left - 1)
            if result:
                path_rec, imp_rec, alt_rec = result
                total_imp = nodes[node]["imp"] + imp_rec
                total_alt = alt_diff + alt_rec
                score = total_imp - weight * total_alt
                if score > best_score:
                    best_score = score
                    best_result = ([node] + path_rec, total_imp, total_alt)

        memo[key] = best_result
        return best_result

    result = recurse(start, max_days)
    return result if result else ([], 0, 0)


def main():
    nodes = load_nodes("dataset/nodes.csv")
    edges = load_edges("dataset/edges.csv", nodes)

    start = input("Start node: ").strip()
    end = input("End node: ").strip()
    days = int(input("Number of days: "))

    print("\nGreedy Approach:")
    path, imp, alt = greedy_path(nodes, edges, start, end, days)
    print(f"Path: {path}")
    print(f"Total Importance: {imp:.4f}, Total Altitude Change: {alt:.2f}")

    print("\nDivide & Conquer Approach:")
    path, imp, alt = dac_path(nodes, edges, start, end, days)
    print(f"Path: {path}")
    print(f"Total Importance: {imp:.4f}, Total Altitude Change: {alt:.2f}")

    print("\nDynamic Programming Approach:")
    path, imp, alt = dp_path(nodes, edges, start, end, days)
    print(f"Path: {path}")
    print(f"Total Importance: {imp:.4f}, Total Altitude Change: {alt:.2f}")


if __name__ == "__main__":
    main()
