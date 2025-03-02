import math
from typing import List, Tuple, Dict


class TrekkingTrail:
    def __init__(self, places: List[Dict]):
        """
        Initialize the trekking trail with a list of places.

        Each place is a dictionary with keys:
        - 'name': Name of the place
        - 'coordinates': Tuple of (latitude, longitude)
        - 'difficulty': Difficulty level of the place (1-10)
        - 'attractions': List of attractions at the place
        """
        self.places = places

    def haversine_distance(
        self, coord1: Tuple[float, float], coord2: Tuple[float, float]
    ) -> float:
        """
        Calculate the great circle distance between two points on the Earth's surface.

        :param coord1: First coordinate (latitude, longitude)
        :param coord2: Second coordinate (latitude, longitude)
        :return: Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
        lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def greedy_trail_selection(
        self, start: str, end: str, total_days: int
    ) -> List[Dict]:
        """
        Greedy algorithm to select trail places.

        Strategy: Always choose the nearest unvisited place that can be reached within remaining days
        and has the most attractions.

        :param start: Starting place name
        :param end: Ending place name
        :param total_days: Total number of days for the trek
        :return: List of selected places for the trail
        """
        # Find start and end places
        start_place = next(p for p in self.places if p["name"] == start)
        end_place = next(p for p in self.places if p["name"] == end)

        # Initialize trail with start place
        trail = [start_place]
        current_place = start_place
        remaining_days = total_days

        while remaining_days > 0 and current_place != end_place:
            # Find unvisited places
            unvisited = [p for p in self.places if p not in trail and p != end_place]

            if not unvisited:
                break

            # Select place based on:
            # 1. Distance from current place
            # 2. Number of attractions
            # 3. Difficulty level
            best_place = max(
                unvisited,
                key=lambda p: (
                    len(p.get("attractions", []))
                    / (
                        self.haversine_distance(
                            current_place["coordinates"], p["coordinates"]
                        )
                        + 1
                    )
                    / p.get("difficulty", 5)
                ),
            )

            # Check if we can reach the place
            distance = self.haversine_distance(
                current_place["coordinates"], best_place["coordinates"]
            )
            travel_days = max(1, math.ceil(distance / 20))  # Assume 20 km per day

            if travel_days <= remaining_days:
                trail.append(best_place)
                current_place = best_place
                remaining_days -= travel_days
            else:
                break

        # Always try to end at the specified end place if possible
        if current_place != end_place:
            trail.append(end_place)

        return trail

    def divide_and_conquer_trail(
        self, start: str, end: str, total_days: int
    ) -> List[Dict]:
        """
        Divide and Conquer approach to trail selection.

        Strategy: Divide the trail into segments, optimize each segment,
        then combine them to create the final trail.

        :param start: Starting place name
        :param end: Ending place name
        :param total_days: Total number of days for the trek
        :return: List of selected places for the trail
        """

        def find_best_segment(places: List[Dict], days: int) -> List[Dict]:
            """
            Find the best segment of places within given days.

            :param places: List of possible places
            :param days: Available days for this segment
            :return: Best segment of places
            """
            if not places or days <= 0:
                return []

            # If only one place, return it
            if len(places) == 1:
                return places

            # Divide places into two halves
            mid = len(places) // 2
            left_half = places[:mid]
            right_half = places[mid:]

            # Recursively find best segments
            left_segment = find_best_segment(left_half, days // 2)
            right_segment = find_best_segment(right_half, days - len(left_segment))

            # Combine segments based on attractions and difficulty
            combined_segment = left_segment + right_segment
            return sorted(
                combined_segment,
                key=lambda p: len(p.get("attractions", [])) / p.get("difficulty", 5),
                reverse=True,
            )[:days]

        # Find start and end places
        start_place = next(p for p in self.places if p["name"] == start)
        end_place = next(p for p in self.places if p["name"] == end)

        # Get places between start and end
        intermediate_places = [
            p for p in self.places if p not in [start_place, end_place]
        ]

        # Select trail using divide and conquer
        trail = [start_place]
        trail.extend(find_best_segment(intermediate_places, total_days - 2))
        trail.append(end_place)

        return trail

    def dynamic_programming_trail(
        self, start: str, end: str, total_days: int
    ) -> List[Dict]:
        """
        Dynamic Programming approach to trail selection.

        Strategy: Build optimal trail by considering all possible combinations
        and maximizing total attractions while minimizing difficulty.

        :param start: Starting place name
        :param end: Ending place name
        :param total_days: Total number of days for the trek
        :return: List of selected places for the trail
        """
        # Find start and end places
        start_place = next(p for p in self.places if p["name"] == start)
        end_place = next(p for p in self.places if p["name"] == end)

        # Get intermediate places
        intermediate_places = [
            p for p in self.places if p not in [start_place, end_place]
        ]

        # Dynamic Programming table
        # dp[i][j] stores the best trail configuration for first i places in j days
        dp = [
            [[] for _ in range(total_days + 1)]
            for _ in range(len(intermediate_places) + 1)
        ]

        # Initialize first row (no places selected)
        for j in range(total_days + 1):
            dp[0][j] = [start_place]

        # Fill DP table
        for i in range(1, len(intermediate_places) + 1):
            for j in range(1, total_days + 1):
                place = intermediate_places[i - 1]

                # Try including or excluding current place
                # Calculate best trail by considering:
                # 1. Number of attractions
                # 2. Difficulty of the place
                # 3. Distance from previous place
                best_trail = dp[i - 1][j].copy()  # Option 1: Exclude current place

                # Option 2: Include current place if possible
                if j > 1:
                    prev_place = (
                        dp[i - 1][j - 1][-1] if dp[i - 1][j - 1] else start_place
                    )
                    distance = self.haversine_distance(
                        prev_place["coordinates"], place["coordinates"]
                    )
                    travel_days = max(1, math.ceil(distance / 20))

                    if travel_days < j:
                        candidate_trail = dp[i - 1][j - travel_days] + [place]

                        # Compare trails based on attractions and difficulty
                        if len(place.get("attractions", [])) / place.get(
                            "difficulty", 5
                        ) > len(best_trail[-1].get("attractions", [])) / best_trail[
                            -1
                        ].get("difficulty", 5):
                            best_trail = candidate_trail

                dp[i][j] = best_trail

        # Add end place to the best trail
        best_trail = dp[len(intermediate_places)][total_days] + [end_place]

        return best_trail


# Example usage
def main():
    # Sample trekking trail places
    places = [
        {
            "name": "Base Camp",
            "coordinates": (27.9881, 86.9250),
            "difficulty": 3,
            "attractions": ["Starting point", "Camp facilities"],
        },
        {
            "name": "Namche Bazaar",
            "coordinates": (27.8044, 86.7214),
            "difficulty": 5,
            "attractions": ["Sherpa culture", "Mountain views", "Local market"],
        },
        {
            "name": "Tengboche Monastery",
            "coordinates": (27.8361, 86.7655),
            "difficulty": 6,
            "attractions": ["Ancient monastery", "Panoramic Himalayan view"],
        },
        {
            "name": "Dingboche",
            "coordinates": (27.9136, 86.8337),
            "difficulty": 7,
            "attractions": ["Acclimatization stop", "Alpine scenery"],
        },
        {
            "name": "Lobuche",
            "coordinates": (27.9406, 86.7922),
            "difficulty": 8,
            "attractions": ["High altitude experience", "Everest views"],
        },
        {
            "name": "Everest Base Camp",
            "coordinates": (28.0029, 86.8530),
            "difficulty": 9,
            "attractions": ["Mount Everest", "Glaciers", "Climbing experience"],
        },
    ]

    # Create TrekkingTrail instance
    trail = TrekkingTrail(places)

    # Demonstrate different algorithms
    print("Greedy Trail Selection:")
    greedy_result = trail.greedy_trail_selection("Base Camp", "Everest Base Camp", 7)
    for place in greedy_result:
        print(f"{place['name']} - Attractions: {place.get('attractions')}")

    print("\nDivide and Conquer Trail:")
    divide_conquer_result = trail.divide_and_conquer_trail(
        "Base Camp", "Everest Base Camp", 7
    )
    for place in divide_conquer_result:
        print(f"{place['name']} - Attractions: {place.get('attractions')}")

    print("\nDynamic Programming Trail:")
    dp_result = trail.dynamic_programming_trail("Base Camp", "Everest Base Camp", 7)
    for place in dp_result:
        print(f"{place['name']} - Attractions: {place.get('attractions')}")


if __name__ == "__main__":
    main()
