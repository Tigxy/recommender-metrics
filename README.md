# Recommender metrics

This is a collection of commonly used recommendation system (RS) metrics. 
As fairness in RS is becoming increasingly important, it is also extended by 
functions to ease computing the differences of RS performances for different 
user groups, e.g., gender.

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

_Notes_:  
*\* Averaging `average precision` and `reciprocal rank` of multiple samples 
leads to `mean average precision (MAP)` and `mean reciprocal rank (MRR)`, respectively, 
which are often used in research.*

## Installation
- Install it as usual with pip:
```python -m pip install git+https://github.com/tigxy/recommender-metrics.git```

- Or from source:
```python -m pip install .```

## Usage metrics

To compute the metrics, simply call them with your model's output, the
true (known) interactions and some cut-off value `k`:
```py
from rmet import ndcg
ndcg(model_output, targets, k=10)
```
Note: `Coverage` does not require the `targets` attribute.

To compute multiple metrics with a single call, check out the `calculte` function,
which accepts a list of metrics to compute:
```py
from rmet import calculate
calculate(["ndcg", "recall"], model_output, targets, k=10, return_aggregated=False)
```

Sample output:
```
{
 'ndcg_aggr': 0.479,
 'recall_aggr': 0.350,
 'ndcg': tensor([0.0000, 0.4693, 0.4693, 1.0000, 0.7039, 0.2346]),
 'recall': tensor([0.0000, 0.2500, 0.3333, 0.6000, 0.6667, 0.2500])
}
```

Here, if `aggregate_results` is set, for each metric the mean of the 
individual users is calculated.

## Usage metric differences for user features

One can also instantiate the `UserFeature` class for some demographic user feature,
such that the performance difference of RS on for different users can be 
evaluated, e.g., for male and female users in the context of gender.

To do so, you first need to specify which feature belongs to which user via the 
`UserGroup` class and then simply call `calculate_for_group` similar to `calculate` above.

```py
from rmet import UserFeature, calculate_for_feature
ug_gender = UserFeature("gender", ["m", "m", "f", "d", "m"])
calculate_for_feature(ug_gender, ["ndcg", "recall"], model_output, targets, k=10)
```

Sample output:

```
{'gender_f': {'ndcg': 0.195, 'recall': 0.125},
 'gender_m': {'ndcg': 0.779, 'recall': 0.733},
 'gender_d': {'ndcg': 0.390, 'recall': 0.458},
 'gender_f-m': {'ndcg': -0.584, 'recall': -0.608},
 'gender_f-d': {'ndcg': -0.195, 'recall': -0.333},
 'gender_m-d': {'ndcg': 0.388, 'recall': 0.275}}
```

## License
MIT License - see the [LICENSE](/LICENSE) file for more details.