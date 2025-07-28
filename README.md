# Recommender metrics

This library is a colletion of common recommender system (RS) evaluation metrics. Moreover, as RS might perform differently for different user groups due to limitations in available data, this library supports the out-of-the-box computations for subsets of users.

## Table of Contents
- [Recommender metrics](#recommender-metrics)
  - [Table of Contents](#table-of-contents)
  - [Metrics Overview](#metrics-overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Single computations](#single-computations)
    - [Multiple metrics and thresholds](#multiple-metrics-and-thresholds)
    - [Computations per user group](#computations-per-user-group)
    - [Batch-wise evaluation](#batch-wise-evaluation)
      - [Overall computation](#overall-computation)
      - [Including user groups](#including-user-groups)
    - [\[Deprecated\] Usage metric differences for user features](#deprecated-usage-metric-differences-for-user-features)
  - [License](#license)

## Metrics Overview

The following metrics are supported (all with the cut-off threshold `k`):
- [DCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Discounted_Cumulative_Gain)
- [nDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)
- [Precision](https://en.wikipedia.org/wiki/Precision_and_recall#Precision)
- [Recall](https://en.wikipedia.org/wiki/Precision_and_recall#Recall)
- [F-score](https://en.wikipedia.org/wiki/F-score#Definition)
- [Average Precision (AP)*](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)
- [Reciprocal Rank (RR)*](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- Hitrate
- Coverage
- Average Rank

This library focuses on efficient metric implementations for PyTorch tensors, NumPy arrays and sparse arrays.

_Notes_:  
*\* Averaging `average precision` and `reciprocal rank` of multiple samples 
leads to `mean average precision (MAP)` and `mean reciprocal rank (MRR)`, respectively, 
which are often used in research.*

## Installation
- Install via pip:
```python -m pip install rmet```

- Or from source:
```python -m pip install .```

## Usage
There are different ways to compute metrics. In the following, we are going to list all of them.

### Single computations

To compute individual metrics, simply import and call them with your model's output (the logits), the true (known) interactions and some cut-off value `k`:
```py
from rmet import ndcg
ndcg(model_output, targets, k=10)
```
Sample output:
```
0.033423
```

Note: `Coverage` does not require the `targets` attribute.

### Multiple metrics and thresholds

You can also call calculate multiple metrics and thresholds efficiently with a single function call. To do so, check out the `calculate` function:
```py
from rmet import calculate

calculate(
    metrics=["ndcg", "recall"], 
    logits=model_output, 
    targets=targets, 
    k=[10, 50],
    return_individual=False,
    flatten_results=True,
)
```

Sample output:
```yaml
{
 'ndcg@10': 0.479,
 'ndcg@50': 0.5,
 'recall@10': 0.350,
 'recall@50': 0.363
}
```

If `return_individual` is set, the metrics are also returned on sample level, e.g., for every user, when possible.

Please check out the functions docstring for the full feature description and its extended functionality.

### Computations per user group

If you want to get insights into the performance of different user groups, e.g., to study differences in recommendation performance based on the users' countries of origin, check out the `calculate_per_group` function:

```py
from rmet import calculate_per_group

# your actual groups as an iterable, e.g., list or pd.Series
group_assignment = ["AT", "DE", "FR", ...] 

calculate_per_group(
    group_name="country",
    group_assignment=group_assignment,
    metrics=["ndcg", "recall"], 
    logits=model_output, 
    targets=targets, 
    k=[10],
    return_individual=False,
    flatten_results=True,
)
```

Sample output:
```yaml
{
 'ndcg@10/country_AT': 0.173,
 'ndcg@10/country_DE': 0.199,
 'ndcg@10/country_FR': 0.239,
 'recall@10/country_AT': 0.282,
 'recall@10/country_DE': 0.301,
 'recall@10/country_FR': 0.357,
}
```

### Batch-wise evaluation
For big datasets and real-world applications, gathering all the logits and targets before computing the recommendation metrics may be too resource-intensive. To simplify calculations in such scenarios, we provide `BatchEvaluator`, a class that evaluates and stores intermediary results.

#### Overall computation

```py
from rmet import BatchEvaluator

# instantiate the evaluator class
batch_evaluator = BatchEvaluator(
    metrics=["ndcg"],
    top_k=[10],
)

# iterate over the batches
for batch in batches:
    user_indices, logits, targets = batch

    # you need to call 'eval_batch' for each batch
    batch_evaluator.eval_batch(
        user_indices=user_indices
        logits=logits,
        targets=targets,
    )

# use 'get_results' to determine the final results
batch_evaluator.get_results()
```
Sample output:
```yaml
{
 'ndcg@10': 0.121,
}
```

#### Including user groups

`BatchEvaluator.eval_batch()` also accepts group assignments as input, which allows the computation of metrics on group and global level. 
```py
from rmet import BatchEvaluator

# instantiate the evaluator class
batch_evaluator = BatchEvaluator(
    metrics=["ndcg"],
    top_k=[10],
)

# iterate over the batches
for batch in batches:
    # batch also returns group_assignments, which is 
    # a mapping from group_name to their values, e.g.,
    # {"country": ["AT", "DE", ...], "gender": ["m", "n", ...]} 
    user_indices, logits, targets, group_assignments = batch

    # you need to call 'eval_batch' for each batch
    batch_evaluator.eval_batch(
        user_indices=user_indices
        logits=logits,
        targets=targets,
        group_assignments=group_assignments,
    )

# use 'get_results' to determine the final results
batch_evaluator.get_results()
```
Sample output:
```yaml
{
 'ndcg@10': 0.121,
 'ndcg@10/country_AT': 0.115,
 'ndcg@10/country_DE': 0.142,
 'ndcg@10/gender_m': 0.087,
 'ndcg@10/gender_f': 0.156,
}
```

### [Deprecated] Usage metric differences for user features

**[NOTE] This feature is deprecated, use `calculate_per_group` and the `BatchEvaluator` with group_assignments instead.**

One can also instantiate the `UserFeature` class for some demographic user feature,
such that the performance difference of RS on for different users can be 
evaluated, e.g., for male and female users in the context of gender.

To do so, you first need to specify which feature belongs to which user via the 
`UserGroup` class and then simply call `calculate_for_group` similar to `calculate` above.

```py
from rmet import UserFeature, calculate_for_feature
ug_gender = UserFeature("gender", ["m", "m", "f", "d", "m"])

calculate_for_feature(
    ug_gender, 
    metrics=["ndcg", "recall"], 
    logits=model_output, 
    targets=targets, 
    k=10,
    return_individual=False,
    flatten_results=True,
)
```

Sample output:

```
{
    'gender_f': {'ndcg@10': 0.195, 'recall@10': 0.125},
    'gender_m': {'ndcg@10': 0.779, 'recall@10': 0.733},
    'gender_d': {'ndcg@10': 0.390, 'recall@10': 0.458},
    'gender_f-m': {'ndcg@10': -0.584, 'recall@10': -0.608},
    'gender_f-d': {'ndcg@10': -0.195, 'recall@10': -0.333},
    'gender_m-d': {'ndcg@10': 0.388, 'recall@10': 0.275}
}
```

## License
MIT License - see the [LICENSE](/LICENSE) file for more details.