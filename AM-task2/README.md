
# Task 2

Labeling scheme to tie application parameters to mode.

## Method

- Jaccard distance:
Measure of dissimilarity between two sets, derived directly from Jaccard
similarity.

- SimHash:
Dimensionality-reduction algorithm that converts large datasets into compact
binary fingerprints. SimHash ensures that similar inputs produce similar
fingerprints.

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
