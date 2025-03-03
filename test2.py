import math
from typing import List, Tuple, Dict, Optional, Union


class TrekkingTrail:
    def __init__(self, places: List[Dict]):
        """
        Initialize the trekking trail with a list of places.

        Each place can be in one of two formats:

        Original format:
        - 'name': Name of the place
        - 'coordinates': Tuple of (latitude, longitude)
        - 'difficulty': Difficulty level of the place (1-10)
        - 'attractions': List of attractions at the place

        New format:
        - 'location_name': Name of the location
        - 'latitude': Latitude coordinate
        - 'longitude': Longitude coordinate
        - 'location_importance': Importance of the location
        - 'altitude': Altitude of the location in meters
        - And other optional fields prefixed with '*'
        """
        # Normalize places data to a consistent format
        self.places = self._normalize_places(places)

    def _normalize_places(self, places: List[Dict]) -> List[Dict]:
        """
        Normalize different data formats into a consistent internal format.

        :param places: List of place dictionaries in different formats
        :return: List of normalized place dictionaries
        """
        normalized_places = []

        for place in places:
            normalized = {}

            # Handle original format
            if "name" in place:
                normalized["name"] = place["name"]
                normalized["latitude"], normalized["longitude"] = place["coordinates"]
                normalized["difficulty"] = place.get("difficulty", 5)
                normalized["attractions"] = place.get("attractions", [])
                normalized["importance"] = 0.5  # Default importance
                normalized["altitude"] = 0  # Default altitude

            # Handle new format
            elif "latitude" in place and "longitude" in place:
                normalized["name"] = place.get(
                    "location_name",
                    f"Location at {place['latitude']}, {place['longitude']}",
                )
                normalized["latitude"] = float(place["latitude"])
                normalized["longitude"] = float(place["longitude"])
                normalized["importance"] = float(place.get("location_importance", 0.5))
                normalized["altitude"] = int(place.get("altitude", 0))

                # Create a virtual difficulty based on altitude (higher = more difficult)
                # Scale from 1-10 based on altitude range
                if "altitude" in place:
                    normalized["difficulty"] = min(
                        10, max(1, int(place["altitude"] / 500))
                    )
                else:
                    normalized["difficulty"] = 5

                # Add attractions based on available fields
                attractions = []
                if place.get("address_village"):
                    attractions.append(f"Village: {place['address_village']}")
                if place.get("address_hamlet"):
                    attractions.append(f"Hamlet: {place['address_hamlet']}")
                normalized["attractions"] = attractions

            # Add the normalized place to our list
            normalized_places.append(normalized)

        return normalized_places

    def haversine_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate the great circle distance between two points on the Earth's surface.

        :param lat1: Latitude of first point
        :param lon1: Longitude of first point
        :param lat2: Latitude of second point
        :param lon2: Longitude of second point
        :return: Distance in kilometers
        """
        R = 6371  # Earth's radius in kilometers

        lat1, lon1 = math.radians(lat1), math.radians(lon1)
        lat2, lon2 = math.radians(lat2), math.radians(lon2)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def calculate_score(self, place1: Dict, place2: Dict) -> float:
        """
        Calculate a score between two places based on:
        1. Location importance (higher is better)
        2. Altitude difference (lower is better)
        3. Distance (lower is better)

        :param place1: First place
        :param place2: Second place
        :return: Score value (higher is better)
        """
        # Calculate distance
        distance = self.haversine_distance(
            place1["latitude"],
            place1["longitude"],
            place2["latitude"],
            place2["longitude"],
        )

        # Calculate altitude difference
        alt_diff = abs(place1.get("altitude", 0) - place2.get("altitude", 0))

        # Get importance
        importance = place2.get("importance", 0.5)

        # Score formula: importance / (distance + 1) / (altitude_difference + 1)
        # The +1 prevents division by zero
        score = importance / (distance + 1) / (alt_diff / 100 + 1)

        return score

    def greedy_trail_selection(
        self, start: str, end: str, total_days: int
    ) -> List[Dict]:
        """
        Greedy algorithm to select trail places.

        Strategy: Always choose the nearest unvisited place that can be reached within remaining days,
        maximizing importance and minimizing altitude differences.

        :param start: Starting place name
        :param end: Ending place name
        :param total_days: Total number of days for the trek
        :return: List of selected places for the trail
        """
        # Find start and end places
        start_place = next((p for p in self.places if p["name"] == start), None)
        end_place = next((p for p in self.places if p["name"] == end), None)

        if not start_place or not end_place:
            raise ValueError(f"Start or end place not found: {start}, {end}")

        # Initialize trail with start place
        trail = [start_place]
        current_place = start_place
        remaining_days = total_days

        while remaining_days > 0 and current_place != end_place:
            # Find unvisited places
            unvisited = [p for p in self.places if p not in trail and p != end_place]

            if not unvisited:
                break

            # Select place based on the calculated score
            best_place = max(
                unvisited, key=lambda p: self.calculate_score(current_place, p)
            )

            # Check if we can reach the place
            distance = self.haversine_distance(
                current_place["latitude"],
                current_place["longitude"],
                best_place["latitude"],
                best_place["longitude"],
            )
            travel_days = max(1, math.ceil(distance / 20))  # Assume 20 km per day

            if travel_days <= remaining_days:
                trail.append(best_place)
                current_place = best_place
                remaining_days -= travel_days
            else:
                break

        # Always try to end at the specified end place if possible
        if current_place != end_place and end_place not in trail:
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
        # Find start and end places
        start_place = next((p for p in self.places if p["name"] == start), None)
        end_place = next((p for p in self.places if p["name"] == end), None)

        if not start_place or not end_place:
            raise ValueError(f"Start or end place not found: {start}, {end}")

        def find_best_segment(
            places: List[Dict], days: int, start_place: Dict
        ) -> List[Dict]:
            """
            Find the best segment of places within given days.

            :param places: List of possible places
            :param days: Available days for this segment
            :param start_place: Starting place for this segment
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
            left_segment = find_best_segment(left_half, days // 2, start_place)

            # Use the last place of left segment as start for right segment
            right_start = left_segment[-1] if left_segment else start_place
            right_segment = find_best_segment(
                right_half, days - len(left_segment), right_start
            )

            # Combine segments
            combined_segment = left_segment + right_segment

            # Sort combined segment by score (importance, altitude, distance)
            if combined_segment:
                curr_place = start_place
                sorted_segment = []

                while combined_segment and len(sorted_segment) < days:
                    best_idx = max(
                        range(len(combined_segment)),
                        key=lambda i: self.calculate_score(
                            curr_place, combined_segment[i]
                        ),
                    )
                    best_place = combined_segment.pop(best_idx)
                    sorted_segment.append(best_place)
                    curr_place = best_place

                return sorted_segment

            return combined_segment

        # Get places between start and end
        intermediate_places = [
            p for p in self.places if p not in [start_place, end_place]
        ]

        # Select trail using divide and conquer
        trail = [start_place]
        trail.extend(
            find_best_segment(intermediate_places, total_days - 2, start_place)
        )
        trail.append(end_place)

        return trail

    def dynamic_programming_trail(
        self, start: str, end: str, total_days: int
    ) -> List[Dict]:
        """
        Dynamic Programming approach to trail selection.

        Strategy: Build optimal trail by considering all possible combinations
        and maximizing location importance while minimizing altitude differences.

        :param start: Starting place name
        :param end: Ending place name
        :param total_days: Total number of days for the trek
        :return: List of selected places for the trail
        """
        # Find start and end places
        start_place = next((p for p in self.places if p["name"] == start), None)
        end_place = next((p for p in self.places if p["name"] == end), None)

        if not start_place or not end_place:
            raise ValueError(f"Start or end place not found: {start}, {end}")

        # Get intermediate places
        intermediate_places = [
            p for p in self.places if p not in [start_place, end_place]
        ]

        # Dynamic Programming table
        # dp[i][j] stores (best_score, trail) for first i places in j days
        dp = [
            [(0, []) for _ in range(total_days + 1)]
            for _ in range(len(intermediate_places) + 1)
        ]

        # Initialize first row (no places selected)
        for j in range(total_days + 1):
            dp[0][j] = (0, [start_place])

        # Fill DP table
        for i in range(1, len(intermediate_places) + 1):
            for j in range(1, total_days + 1):
                place = intermediate_places[i - 1]

                # Option 1: Exclude current place
                best_score, best_trail = dp[i - 1][j]

                # Option 2: Include current place if possible
                if j > 1:
                    for k in range(1, j):
                        prev_score, prev_trail = dp[i - 1][j - k]
                        prev_place = prev_trail[-1] if prev_trail else start_place

                        distance = self.haversine_distance(
                            prev_place["latitude"],
                            prev_place["longitude"],
                            place["latitude"],
                            place["longitude"],
                        )
                        travel_days = max(1, math.ceil(distance / 20))

                        if travel_days <= k:
                            # Calculate new score based on importance and altitude difference
                            place_score = self.calculate_score(prev_place, place)
                            new_score = prev_score + place_score

                            if new_score > best_score:
                                best_score = new_score
                                best_trail = prev_trail + [place]

                dp[i][j] = (best_score, best_trail)

        # Get the best trail
        _, best_trail = dp[len(intermediate_places)][total_days]

        # Add end place if not already included
        if best_trail and best_trail[-1] != end_place:
            best_trail.append(end_place)

        return best_trail


# Example usage
def main():
    # Sample data in new format
    places = [
        {
            "latitude": 27.7878,
            "longitude": 85.8998333333333,
            "location_name": "Arniko Highway",
            "location_importance": 0.0533833348592354,
            "altitude": 857,
        },
        {
            "latitude": 27.7927,
            "longitude": 85.9255,
            "location_name": "Bahrabise - Karthali - Dolangsa Road",
            "location_importance": 0.0533833348592354,
            "altitude": 1538,
            "address_village": "Karthali",
            "address_hamlet": "Drumtali",
        },
        {
            "latitude": 27.7965,
            "longitude": 85.939,
            "location_name": "Bahrabise - Karthali - Dolangsa Road",
            "location_importance": 0.0533833348592354,
            "altitude": 1723,
            "address_village": "Karthali",
            "address_hamlet": "Drumtali",
        },
        {
            "latitude": 27.7902,
            "longitude": 85.9429,
            "location_name": "Budepa Road",
            "location_importance": 0.0533833348592354,
            "altitude": 1571,
            "address_village": "Karthali",
            "address_hamlet": "Budepa",
        },
    ]

    # Create TrekkingTrail instance
    trail = TrekkingTrail(places)

    # Demonstrate different algorithms
    print("Greedy Trail Selection:")
    start_location = "Arniko Highway"
    end_location = "Budepa Road"
    greedy_result = trail.greedy_trail_selection(start_location, end_location, 4)
    for place in greedy_result:
        print(
            f"{place['name']} - Altitude: {place.get('altitude')}m, Importance: {place.get('importance')}"
        )

    print("\nDivide and Conquer Trail:")
    divide_conquer_result = trail.divide_and_conquer_trail(
        start_location, end_location, 4
    )
    for place in divide_conquer_result:
        print(
            f"{place['name']} - Altitude: {place.get('altitude')}m, Importance: {place.get('importance')}"
        )

    print("\nDynamic Programming Trail:")
    dp_result = trail.dynamic_programming_trail(start_location, end_location, 4)
    for place in dp_result:
        print(
            f"{place['name']} - Altitude: {place.get('altitude')}m, Importance: {place.get('importance')}"
        )


if __name__ == "__main__":
    main()
