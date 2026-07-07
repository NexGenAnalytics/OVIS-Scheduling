
# Task 2

Labeling scheme to tie application parameters to mode.

## Method

- Normalize into sets and Jaccard distance:
Measure of dissimilarity between two sets, derived directly from Jaccard
similarity.

- SimHash and Hamming Distance:
Dimensionality-reduction algorithm that converts large datasets into compact
binary fingerprints. SimHash ensures that similar inputs produce similar
fingerprints. The Hamming distance is a metric used to measure the dissimilarity
between two data sequences (like binary) of equal length.

## Setup

```sh
pwd # [...]/AM-task2

python3 -m venv .venv
source .venv/bin/activate # for Linux and Mac
.\.venv\Scripts\activate  # for Windows

(.venv) pip install -r requirements.txt

(.venv) python script.py [path1] [path2]
# example: python script.py ./configs/DarmaPerformance.txt ./configs/PressioDebug.txt

(.venv) deactivate
```
