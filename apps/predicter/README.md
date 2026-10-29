# `predicter/models.py`

- **Dummy:** A simple baseline that ignores the input features and always
predicts a fixed value, such as the mean target value. It helps assess whether
more complex models add value.

```mermaid
flowchart LR
    A["Past durations: 8, 10, 12"] --> B["Calculate mean: 10"]
    B --> C["Predict 10 for every new task"]
```

- **Ridge:** A linear regression model with a penalty that shrinks coefficients,
reducing overfitting and improving stability when features are correlated.

```mermaid
flowchart LR
    A["Task features"] --> B["Multiply by learned weights"]
    B --> C["Add results + intercept"]
    C --> D["Predicted duration"]
    E["Training penalizes large weights"] -.-> B
```

- **MLP:** A multilayer perceptron, or neural network, that learns nonlinear
relationships through layers of interconnected neurons.

```mermaid
flowchart LR
    A["Feature 1"] --> C(("Neuron"))
    A --> D(("Neuron"))
    B["Feature 2"] --> C
    B --> D
    C --> E(("Neuron"))
    D --> E
    E --> F["Predicted duration"]
```

- **Random forest:** An ensemble of decision trees trained on different random
samples of the data. Averaging their predictions improves robustness and
reduces overfitting.

```mermaid
flowchart LR
    A["Task features"] --> B["Tree 1 → 8"]
    A --> C["Tree 2 → 10"]
    A --> D["Tree 3 → 12"]
    B --> E["Average → 10"]
    C --> E
    D --> E
```

- **Extra trees:** An ensemble similar to a random forest, but with more
randomness when choosing tree splits. This often makes training faster and
reduces prediction variance.

```mermaid
flowchart LR
    A["Task features"] --> B["Randomized tree 1 → 7"]
    A --> C["Randomized tree 2 → 10"]
    A --> D["Randomized tree 3 → 13"]
    B --> E["Average → 10"]
    C --> E
    D --> E
```

- **Hist GB:** Histogram-based gradient boosting builds trees sequentially, with
each tree correcting errors from the previous ones. Grouping feature values into
bins makes training efficient.

```mermaid
flowchart LR
    A["Starting prediction: 10"] --> B["Tree 1 correction: +3"]
    B --> C["Tree 2 correction: −1"]
    C --> D["Final prediction: 12"]
```

# Architecture

```
*cli.py*
-> load_datas
---> parse_inputdeck / *parsing.py*
---> filter_metric / *visualizer.metrics*
---> max_value / *visualizer.metrics*
-> split_set
-> prepare_dataset / *prepare.py*
---> define_deck / *decker.py*
-> run_benchmark / *benchmark.py*
---> evaluate_predictions / *evaluation.py*

```
