## What this folder help with

- Compare 2 input configuration.
- Populate a database from a folder of input configuration files.
- Search inside a database the closest job.
- Simple prediction based on the execution command.

## Comparison methods

- Normalize into sets then Jaccard distance:
Measure of dissimilarity between two sets, derived directly from Jaccard
similarity.

- SimHash then Hamming Distance:
Dimensionality-reduction algorithm that converts large datasets into compact
binary fingerprints. SimHash ensures that similar inputs produce similar
fingerprints. The Hamming distance is a metric used to measure the dissimilarity
between two data sequences (like binary) of equal length.

## Prediction methods

- From execution command.
- From tensor/matrix given.

## Setup and usage

```sh
pwd # [...]/AM-task2

python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate  # for Windows

(.venv) pip install -r requirements.txt

(.venv) python src/compare.py [config_file_path1] [config_file_path2]
# Compute Jaccard and Hamming distance
# example: python src/compare.py configs/DarmaPerformance.txt configs/PressioDebug.txt
# result: display in terminal

(.venv) python src/prediction.py [command_file_path]
# Predict execution values from command
# example: python src/prediction.py commands/KokkosCuda.txt
# result: display in terminal

(.venv) python src/save.py [config_folder_path]
# Create and populate database
# example: python src/save.py configs/
# result: file db.json

(.venv) python src/search.py [db_path] [config_file_path]
# Find the closest job
# example: python src/search.py db.json configs/MyConfig.txt
# result: display in terminal

(.venv) deactivate
```

## Notes

- `commands/`: commands that effectively run code (needed to measure utilization/usage/occupancy/network).
- `configs/`: CMake flags only choose what gets built.
