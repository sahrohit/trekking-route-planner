import csv

# Constants for trekking parameters
HOURS_PER_DAY = 6  # Person can trek 6 hours a day
AVG_SPEED_MPS = 1  # Average trekking speed in meters per second
MAX_DISTANCE_PER_DAY = (
    HOURS_PER_DAY * 3600 * AVG_SPEED_MPS
)  # Maximum distance in meters per day


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
            # Only add edges that can be traversed in a day
            if dist <= MAX_DISTANCE_PER_DAY:
                edges.setdefault(source, []).append((target, dist, alt_diff))
                edges.setdefault(target, []).append((source, dist, alt_diff))
    return edges


def score_path(
    path, nodes, imp_weight=0.4, alt_weight=0.4, days_weight=0.2, max_days=None
):
    """
    Score a path based on three criteria:
    1. Days used vs max days (maximize days used but stay within limit)
    2. Average importance (maximize)
    3. Total altitude change (minimize)

    Returns a tuple of (score, days_used, avg_importance, alt_change)
    """
    if not path or len(path) < 2:
        return -float("inf"), 0, 0, 0

    # Calculate metrics
    days_used = len(path) - 1
    total_imp = sum(nodes[node]["imp"] for node in path)
    avg_imp = total_imp / len(path)

    # Calculate total altitude change (from sequential nodes in path)
    total_alt = 0
    for i in range(len(path) - 1):
        alt1 = nodes[path[i]]["alt"]
        alt2 = nodes[path[i + 1]]["alt"]
        total_alt += abs(alt2 - alt1)

    # Days score: maximize days used (within limit)
    days_score = (
        days_used / max_days if max_days and days_used <= max_days else -float("inf")
    )

    # Normalize altitude change (lower is better) - assume 1000m is a reference point
    alt_penalty = total_alt / (days_used * 1000) if days_used > 0 else 0

    # Combined score: maximize days used & importance, minimize altitude change
    score = days_weight * days_score + imp_weight * avg_imp - alt_weight * alt_penalty

    return score, days_used, avg_imp, total_alt


def greedy_path(
    nodes, edges, start, end, max_days, imp_weight=0.4, alt_weight=0.4, days_weight=0.2
):
    """
    Greedy approach with backtracking: choose the neighbor that maximizes our combined score.
    If we hit a dead end, backtrack and try the next best option.
    """
    # Stack to keep track of our decisions for backtracking
    # Each entry is (node, [neighbors_to_try], visited_set)
    decision_stack = [(start, edges.get(start, []).copy(), {start})]
    path = [start]

    while decision_stack and len(path) - 1 < max_days:
        current, neighbors, visited = decision_stack[-1]

        # Check if we've reached the destination
        if current == end:
            break

        # Find the best unvisited neighbor
        best_next = None
        best_score = float("-inf")
        best_neighbor_info = None
        remaining_neighbors = []

        for neighbor_info in neighbors:
            nbr, dist, alt_diff = neighbor_info
            if nbr in visited:
                continue

            # Try this neighbor
            trial_path = path + [nbr]
            score, _, _, _ = score_path(
                trial_path, nodes, imp_weight, alt_weight, days_weight, max_days
            )

            if score > best_score:
                best_score = score
                best_next = nbr
                best_neighbor_info = neighbor_info

            remaining_neighbors.append(neighbor_info)

        # Remove the best neighbor from remaining options
        if best_neighbor_info in remaining_neighbors:
            remaining_neighbors.remove(best_neighbor_info)

        # If we found a valid next step
        if best_next:
            # Update the current node's remaining neighbors
            decision_stack[-1] = (current, remaining_neighbors, visited)

            # Add the new node to our path
            path.append(best_next)
            new_visited = visited.copy()
            new_visited.add(best_next)

            # Push this decision to our stack
            decision_stack.append(
                (best_next, edges.get(best_next, []).copy(), new_visited)
            )
        else:
            # No valid neighbors, need to backtrack
            decision_stack.pop()  # Remove current dead-end node
            if path:  # Make sure path isn't empty before popping
                path.pop()  # Remove from path as well

    # Calculate final metrics
    if path and path[-1] == end:
        score, days_used, avg_imp, total_alt = score_path(
            path, nodes, imp_weight, alt_weight, days_weight, max_days
        )
        total_imp = sum(nodes[node]["imp"] for node in path)
        return path, total_imp, total_alt
    else:
        # Couldn't reach destination within constraints
        return [], 0, 0


def dac_path(
    nodes, edges, start, end, max_days, imp_weight=0.4, alt_weight=0.4, days_weight=0.2
):
    """
    Divide and conquer (brute-force) recursive search without memoization.
    """

    def recurse(node, days_left, visited):
        if days_left < 0:
            return None

        # If we've reached the destination
        if node == end:
            path = [end]
            return path, nodes[end]["imp"], 0

        best_score = float("-inf")
        best_result = None

        for nbr, dist, alt_diff in edges.get(node, []):
            if nbr in visited:
                continue

            result = recurse(nbr, days_left - 1, visited | {nbr})
            if result:
                path_rec, imp_rec, alt_rec = result

                # Calculate metrics for this path
                trial_path = [node] + path_rec
                score, _, _, _ = score_path(
                    trial_path, nodes, imp_weight, alt_weight, days_weight, max_days
                )

                total_imp = nodes[node]["imp"] + imp_rec
                total_alt = alt_diff + alt_rec

                if score > best_score:
                    best_score = score
                    best_result = ([node] + path_rec, total_imp, total_alt)

        return best_result

    result = recurse(start, max_days, {start})
    return result if result else ([], 0, 0)


def dp_path(
    nodes, edges, start, end, max_days, imp_weight=0.4, alt_weight=0.4, days_weight=0.2
):
    """
    Dynamic programming with memoization over (node, days_left).
    Optimizes for our three criteria.
    """
    memo = {}

    def recurse(node, days_left):
        key = (node, days_left)
        if key in memo:
            return memo[key]

        # Base case - at destination
        if node == end:
            memo[key] = ([end], nodes[end]["imp"], 0)
            return memo[key]

        # Out of days
        if days_left <= 0:
            memo[key] = None
            return None

        best_score = float("-inf")
        best_result = None

        for nbr, dist, alt_diff in edges.get(node, []):
            result = recurse(nbr, days_left - 1)
            if result:
                path_rec, imp_rec, alt_rec = result

                # Calculate total metrics
                total_imp = nodes[node]["imp"] + imp_rec
                total_alt = alt_diff + alt_rec

                # Calculate score for this path
                trial_path = [node] + path_rec
                score, _, _, _ = score_path(
                    trial_path, nodes, imp_weight, alt_weight, days_weight, max_days
                )

                if score > best_score:
                    best_score = score
                    best_result = ([node] + path_rec, total_imp, total_alt)

        memo[key] = best_result
        return best_result

    result = recurse(start, max_days)
    return result if result else ([], 0, 0)


def write_results_to_csv(filename, results):
    """
    Write the path finding results to a CSV file.
    Results should be a dictionary with algorithm names as keys and
    (path, importance, altitude_change) tuples as values.
    """
    with open(filename, "w", newline="") as csvfile:
        # Write header row
        fieldnames = [
            "algorithm",
            "path",
            "total_importance",
            "average_importance",
            "total_altitude_change",
            "path_length",
            "days_used",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Write results for each algorithm
        for algorithm, (path, imp, alt) in results.items():
            days_used = len(path) - 1  # Number of edges = number of days
            avg_imp = imp / len(path) if len(path) > 0 else 0
            writer.writerow(
                {
                    "algorithm": algorithm,
                    "path": ",".join(path),
                    "total_importance": f"{imp:.4f}",
                    "average_importance": f"{avg_imp:.4f}",
                    "total_altitude_change": f"{alt:.2f}",
                    "path_length": len(path),
                    "days_used": days_used,
                }
            )


def main():
    nodes = load_nodes("dataset/nodes.csv")
    edges = load_edges("dataset/edges.csv", nodes)

    start = input("Start node: ").strip()
    end = input("End node: ").strip()
    max_days = int(input("Maximum number of days: "))

    # Dictionary to store results
    results = {}

    print("\nGreedy Approach:")
    path, imp, alt = greedy_path(nodes, edges, start, end, max_days)
    results["greedy"] = (path, imp, alt)
    # Calculate metrics for reporting
    days_used = len(path) - 1 if len(path) > 1 else 0
    avg_imp = imp / len(path) if len(path) > 0 else 0
    print(f"Path: {path}")
    print(f"Days used: {days_used}/{max_days}")
    print(f"Total Importance: {imp:.4f}, Average Importance: {avg_imp:.4f}")
    print(f"Total Altitude Change: {alt:.2f}")

    print("\nDivide & Conquer Approach:")
    path, imp, alt = dac_path(nodes, edges, start, end, max_days)
    results["divide_and_conquer"] = (path, imp, alt)
    # Calculate metrics for reporting
    days_used = len(path) - 1 if len(path) > 1 else 0
    avg_imp = imp / len(path) if len(path) > 0 else 0
    print(f"Path: {path}")
    print(f"Days used: {days_used}/{max_days}")
    print(f"Total Importance: {imp:.4f}, Average Importance: {avg_imp:.4f}")
    print(f"Total Altitude Change: {alt:.2f}")

    print("\nDynamic Programming Approach:")
    path, imp, alt = dp_path(nodes, edges, start, end, max_days)
    results["dynamic_programming"] = (path, imp, alt)
    # Calculate metrics for reporting
    days_used = len(path) - 1 if len(path) > 1 else 0
    avg_imp = imp / len(path) if len(path) > 0 else 0
    print(f"Path: {path}")
    print(f"Days used: {days_used}/{max_days}")
    print(f"Total Importance: {imp:.4f}, Average Importance: {avg_imp:.4f}")
    print(f"Total Altitude Change: {alt:.2f}")

    # Write results to CSV
    output_filename = f"results_{start}_to_{end}_{max_days}days.csv"
    write_results_to_csv(output_filename, results)
    print(f"\nResults written to {output_filename}")

    # Save the path data for plotting
    path_data_filename = f"dataset/result/path_data_{start}_to_{end}_{max_days}days.csv"
    with open(path_data_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["algorithm", "node", "latitude", "longitude", "altitude", "importance"]
        )

        for algorithm, (path, imp, alt) in results.items():
            for node_name in path:
                node_data = nodes[node_name]
                writer.writerow(
                    [
                        algorithm,
                        node_name,
                        node_data["lat"],
                        node_data["lon"],
                        node_data["alt"],
                        node_data["imp"],
                    ]
                )

    print(f"Path data for plotting written to {path_data_filename}")


if __name__ == "__main__":
    main()
