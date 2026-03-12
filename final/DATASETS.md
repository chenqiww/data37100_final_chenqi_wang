# Approved Datasets

To keep projects within the expected time budget (\~20 hours), choose
from the datasets below unless approved otherwise.

## Tier 1 --- Recommended

### MNIST

``` python
from torchvision import datasets
datasets.MNIST(root="./data/bigdata/MNIST", download=True)
```

Pros:

-   very fast training\
-   stable\
-   ideal for controlled experiments

------------------------------------------------------------------------

### CIFAR-10

``` python
datasets.CIFAR10(root="./data/bigdata/CIFAR10", download=True)
```

Pros:

-   richer structure\
-   still manageable\
-   widely studied

------------------------------------------------------------------------

## Tier 2 --- Optional

### Fashion-MNIST

Good for representation comparison and diversity analysis.

------------------------------------------------------------------------

## Storage location

Default:

    ./data/bigdata/

When in doubt, use MNIST.
