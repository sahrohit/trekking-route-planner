# Trekking Route Planner

Trekking Route Planning for Nepali Mountain Trails based on Time, Distance and Difficulty

## Description

Nepal, a country known for its stunning mountain landscapes, has numerous trekking trails with their own challenges. However, finding the optimal trekking route while minimizing difficulty and balancing distance and available time remains a significant challenge for trekkers. Our idea is to develop a route planning system that suggests the easiest and the most efficient route from starting point to destination. This system will maximize the number of possible key locations while ensuring the trek remains feasible within the allotted number of days.

## Project Structure

```
trekking-route-planner/
│
├── dataset/               # Contains all datasets
│   ├── final/             # Finalized Data Files
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
│
├── main.py                # Main application script
├── notebook.py            # Experimental notebook-style script
└── pyproject.toml         # Project dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- uv (Python package and project manager)
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/sahrohit/trekking-route-planner.git
   cd trekking-route-planner
   ```

2. Install `uv` if you dont have it installed and then. Create a virtual environment (recommended):
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
      uv sync
   ```

## Usage

### Running the Main Application

The `main.py` script contains the primary functionality for processing and analyzing UV data:

```bash
uv run main.py
```

### Running Experiments

The `notebook.py` file provides a Jupyter-like environment for experimentation:

This script allows for interactive experimentation with different parameters and visualization techniques.

<!-- ## Data Format

### Input Data

The input CSV files should have the following format:

```
timestamp,location,uv_index,temperature,cloud_cover
2023-01-01 12:00:00,City A,7.5,28.3,0.2
...
```

### Output Data

The processed data includes additional calculated fields:

```
timestamp,location,uv_index,temperature,cloud_cover,risk_level,exposure_time
2023-01-01 12:00:00,City A,7.5,28.3,0.2,MODERATE,40
...
``` -->

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

<!-- ## Acknowledgments

- List any libraries, datasets, or resources that you used or were inspired by
- Credit collaborators or institutions -->