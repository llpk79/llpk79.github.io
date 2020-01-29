---
layout: post
title: Multi-label classification for the dice game Ten Thousand.
subtitle: A fun, and waaaay over-engineered solution to creating a computer player for the game, the logic for which is almost entirely implemented in the second cell of this notebook.
image: /img/dice.jpg
---

Multi-label classification where n-features == n-labels sounds like 
a fun challenge aside from applying to my hobby project, let's see
how we do.

Connect to the rest of this project, and others, on [github.](https://github.com/llpk79)

**1 - 2hr approximate notebook runtime.**

Import packages.

```python
import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations_with_replacement as combos
from itertools import permutations as perms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score, label_ranking_average_precision_score, average_precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

```

Establish game rules for making labels.


```python
scoring_rules = [[100, 200, 1000, 2000, 4000, 5000],
                 [0, 0, 200, 400, 800, 5000],
                 [0, 0, 300, 600, 1200, 5000],
                 [0, 0, 400, 800, 1600, 5000],
                 [50, 100, 500, 1000, 2000, 5000],
                 [0, 0, 600, 1200, 2400, 5000]
                 ]


def is_three_pair(roll):
    """Return true if roll contains three pairs.
    
    :var roll: list - A roll of 1 - 6 dice.
    """
    roll = sorted(roll)
    return (len(roll) == 6 and roll[0] == roll[1] and
            roll[2] == roll[3] and roll[4] == roll[5])


def is_straight(roll):
    """Return true if roll contains dice 1 - 6.

    :var roll: list - A roll of 1 - 6 dice.
    """
    return sorted(roll) == list(range(1, 7))


def score_all():
    """Return a list of floats == 1.0."""
    return [1.] * 6


def make_labels(roll):
    """Returns a label for each roll.
    
    :var roll: list - A roll of 1 - 6 dice.
    """
    counts = Counter(roll)
    if is_three_pair(roll) and (sum(scoring_rules[die - 1][count - 1] for die, count in counts.items()) < 1500):
        choice = score_all()
    elif is_straight(roll):
        choice = score_all()
    else:
        picks = set()
        for die, count in counts.items():
            if scoring_rules[die - 1][count - 1] > 0:
                picks.add(die)
        choice = [0.] * 6
        for i, x in enumerate(roll):
            if x in picks:
                choice[i] = 1.
    return np.array(choice)

```

Make a bunch of 6-dice throws.


```python
def make_some_features(clip):
    """Creat a set of outcomes for rolling 6 dice.
    
    Find combinations of 6 dice, then permute them.
    
    :var clip: keep only combinations where its index % clip = 0
    """
    features = set()
    numbers = list(range(1, 7))
    combinations = (combo for combo in combos(numbers, 6))
    for i, comb in enumerate(combinations):
        if i % clip == 0:  # Keeping size reasonable
            for perm in perms(comb):
                features.add(perm)
    return features
```

Make arrays of throws and corresponding labels.


```python
features = make_some_features(2)

all_features = np.array([np.array(feature) for feature in features])

all_labels = np.array([make_labels(feature) for feature in all_features])

all_features.shape, all_labels.shape
```




    ((23114, 6), (23114, 6))


Create a DataFrame.


```python
def create_dataset(features, labels):
    """Create a DataFrame for dice throws and their labels.
    
    A column for each die in a roll and a label for each die.
    
    :var features: np.array of rolls
    :var labels: np.array of labels
    """
    # DataFrame for features.
    data = {str(i): features[:,i] for i in range(6)}
    dataset = pd.DataFrame(data)

    # DataFrame for labels.
    label = {'{}_l'.format(i): labels[:,i] for i in range(6)}
    label_df = pd.DataFrame(label)
    
    # Stick em together.
    df = pd.concat([dataset, label_df], axis=1, sort=False)
    return df
```



```python
df = create_dataset(all_features, all_labels)
```


```python
df.sample(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>0_l</th>
      <th>1_l</th>
      <th>2_l</th>
      <th>3_l</th>
      <th>4_l</th>
      <th>5_l</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21365</th>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>20612</th>
      <td>6</td>
      <td>5</td>
      <td>2</td>
      <td>6</td>
      <td>5</td>
      <td>5</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14333</th>
      <td>3</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>5416</th>
      <td>4</td>
      <td>2</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>429</th>
      <td>6</td>
      <td>4</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>4</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1490</th>
      <td>5</td>
      <td>6</td>
      <td>6</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>11539</th>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>5</td>
      <td>3</td>
      <td>6</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9165</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
      <td>4</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12548</th>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>6</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13071</th>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>4</td>
      <td>6</td>
      <td>4</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Separate X and y sets and split into training and test sets.


```python
X = df[['0', '1', '2', '3', '4', '5']]
y = df[['0_l', '1_l', '2_l', '3_l', '4_l', '5_l']]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True)
```


```python
X_train.shape, y_train.shape
```




    ((17335, 6), (17335, 6))




```python
X_test.shape, y_test.shape
```




    ((5779, 6), (5779, 6))



Extra Trees with hyperparameters chosen from earlier cross validations.


```python
extra = ExtraTreesClassifier(bootstrap=True,
                             oob_score=True,
                             n_jobs=-1,
                             n_estimators=2250)
```

Cross validation with grid search on min_sample_split and max_depth.


```python
params = {'min_samples_split': [4, 5, 6],
          'max_depth': [27, 30, 33]}
grid = GridSearchCV(extra,
                    param_grid=params,
                    scoring='average_precision',
                    n_jobs=-1,
                    cv=5,
                    verbose=1)
grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
```

    Fitting 5 folds for each of 9 candidates, totalling 45 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  45 out of  45 | elapsed: 44.4min finished

    ({'max_depth': 30, 'min_samples_split': 6}, 0.9759599000677779)



Refine n_estimators with grid search.


```python
params = {'n_estimators': [1250, 1500, 1750, 2000, 2250, 2500]}
grid = GridSearchCV(grid.best_estimator_,
                    param_grid=params,
                    scoring='average_precision',
                    n_jobs=-1,
                    cv=5,
                    verbose=1)
grid.fit(X_train, y_train)
grid.best_params_, grid.best_score_
```

    Fitting 5 folds for each of 6 candidates, totalling 30 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  30 out of  30 | elapsed: 29.8min finished

    ({'n_estimators': 2000}, 0.9759340084764407)


Calculate metrics from GridSearchCV best model.

```python
best = grid.best_estimator_
```

Make predictions for each test roll.

```python
y_pred = np.array([best.predict([test])[0] for test in X_test.values])
```

Calculate f1 score across all labels for each instance.

```python
f1_score(y_test, y_pred, average='samples')
```

    0.9027160962703444


Calculate precision across each label instance.

```python
label_ranking_average_precision_score(y_test, y_pred)
```




    0.9735207456115055


Calculate average precision.

```python
average_precision_score(y_test, y_pred)
```




    0.909635794804136



Examine individual predictions at the standard 0.5 probability threshold, and at different thresholds.


```python
def test_model_pred(model, threshold=0.475, samples=25):
    """Get random sample of rolls from X_test and make predictions.
    
    Compare prediction precision with probability > 0.5 positive label with
    positive label at other thresholds by adjusting threshold.
    
    Print number of samples.
    
    :var threshold: float
    :var samples: int
    """
    for test in X_test.sample(samples).values:
        
        print(test)
        
        # Create ground truth label.
        true = make_labels(test).astype(int)
        print(true)
        
        # Raw probability predictions.
        pred_proba = np.array([round(y[0][1], 3) for y in model.predict_proba([list(test)])])
        print(pred_proba)
        
        # Predict 1 if probability > 0.5.
        pred = (pred_proba > 0.5).astype(int)
        print(pred)
        
        # Predict 1 if probability > threshold.
        pred_thresh = (pred_proba > threshold).astype(int)
        print(pred_thresh)
        
        result = 'Nailed it' if list(true) == list(pred) else 'Nuts'
        print(result)
        result = 'Nailed it' if list(true) == list(pred_thresh) else 'Nuts'
        print(result)
        print()
```

We are looking to move a prediction from 'Nuts' to 'Nailed it' 
by choosing a better probability threshold, but also avoid doing 
the opposite.


```python
test_model_pred(best, threshold=.475, samples=40)
```

    
    [1 6 2 2 6 2]
    [1 0 1 1 0 1]
    [0.948 0.33  0.461 0.435 0.328 0.441]
    [1 0 0 0 0 0]
    [1 0 0 0 0 0]
    Nuts
    Nuts
    
    [2 2 3 6 2 1]
    [1 1 0 0 1 1]
    [0.493 0.533 0.314 0.097 0.526 0.967]
    [0 1 0 0 1 1]
    [1 1 0 0 1 1]
    Nuts
    Nailed it
    
    [3 4 6 3 3 1]
    [1 0 0 1 1 1]
    [0.438 0.338 0.096 0.446 0.433 0.922]
    [0 0 0 0 0 1]
    [0 0 0 0 0 1]
    Nuts
    Nuts
    
    [5 1 1 2 5 6]
    [1 1 1 0 1 0]
    [0.755 0.843 0.85  0.257 0.755 0.176]
    [1 1 1 0 1 0]
    [1 1 1 0 1 0]
    Nailed it
    Nailed it
    
    [2 3 1 6 2 3]
    [0 0 1 0 0 0]
    [0.461 0.316 0.948 0.107 0.408 0.379]
    [0 0 1 0 0 0]
    [0 0 1 0 0 0]
    Nailed it
    Nailed it
    
    [2 6 2 1 4 6]
    [0 0 0 1 0 0]
    [0.286 0.264 0.248 0.936 0.239 0.266]
    [0 0 0 1 0 0]
    [0 0 0 1 0 0]
    Nailed it
    Nailed it
    
    [6 5 3 3 6 4]
    [0 1 0 0 0 0]
    [0.297 0.916 0.339 0.331 0.293 0.322]
    [0 1 0 0 0 0]
    [0 1 0 0 0 0]
    Nailed it
    Nailed it
    
    [3 3 5 1 5 1]
    [1 1 1 1 1 1]
    [0.195 0.186 0.778 0.877 0.766 0.88 ]
    [0 0 1 1 1 1]
    [0 0 1 1 1 1]
    Nuts
    Nuts
    
    [2 2 5 4 3 2]
    [1 1 1 0 0 1]
    [0.5   0.499 0.857 0.189 0.292 0.491]
    [0 0 1 0 0 0]
    [1 1 1 0 0 1]
    Nuts
    Nailed it
    
    [1 3 6 4 4 5]
    [1 0 0 0 0 1]
    [0.863 0.191 0.13  0.28  0.315 0.833]
    [1 0 0 0 0 1]
    [1 0 0 0 0 1]
    Nailed it
    Nailed it
    
    [1 3 2 3 5 5]
    [1 0 0 0 1 1]
    [0.93  0.186 0.213 0.197 0.772 0.769]
    [1 0 0 0 1 1]
    [1 0 0 0 1 1]
    Nailed it
    Nailed it
    
    [3 5 5 2 6 6]
    [0 1 1 0 0 0]
    [0.26  0.783 0.806 0.277 0.358 0.344]
    [0 1 1 0 0 0]
    [0 1 1 0 0 0]
    Nailed it
    Nailed it
    
    [1 3 4 3 6 4]
    [1 0 0 0 0 0]
    [0.879 0.397 0.447 0.38  0.138 0.526]
    [1 0 0 0 0 1]
    [1 0 0 0 0 1]
    Nuts
    Nuts
    
    [5 2 5 2 3 4]
    [1 0 1 0 0 0]
    [0.833 0.437 0.811 0.443 0.338 0.314]
    [1 0 1 0 0 0]
    [1 0 1 0 0 0]
    Nailed it
    Nailed it
    
    [4 1 5 5 4 2]
    [0 1 1 1 0 0]
    [0.247 0.933 0.812 0.797 0.236 0.188]
    [0 1 1 1 0 0]
    [0 1 1 1 0 0]
    Nailed it
    Nailed it
    
    [5 3 4 6 4 3]
    [1 0 0 0 0 0]
    [0.891 0.349 0.361 0.115 0.371 0.393]
    [1 0 0 0 0 0]
    [1 0 0 0 0 0]
    Nailed it
    Nailed it
    
    [1 2 6 1 2 3]
    [1 0 0 1 0 0]
    [0.9   0.458 0.143 0.877 0.436 0.329]
    [1 0 0 1 0 0]
    [1 0 0 1 0 0]
    Nailed it
    Nailed it
    
    [4 6 6 2 3 2]
    [0 0 0 0 0 0]
    [0.498 0.421 0.426 0.453 0.411 0.487]
    [0 0 0 0 0 0]
    [1 0 0 0 0 1]
    Nailed it
    Nuts
    
    [4 6 3 5 2 3]
    [0 0 0 1 0 0]
    [0.208 0.106 0.286 0.846 0.241 0.286]
    [0 0 0 1 0 0]
    [0 0 0 1 0 0]
    Nailed it
    Nailed it
    
    [3 4 4 4 3 6]
    [0 1 1 1 0 0]
    [0.374 0.473 0.523 0.517 0.385 0.1  ]
    [0 0 1 1 0 0]
    [0 0 1 1 0 0]
    Nuts
    Nuts
    
    [1 2 4 3 5 1]
    [1 0 0 0 1 1]
    [0.89  0.184 0.146 0.162 0.751 0.894]
    [1 0 0 0 1 1]
    [1 0 0 0 1 1]
    Nailed it
    Nailed it
    
    [6 3 4 6 2 1]
    [0 0 0 0 0 1]
    [0.258 0.224 0.272 0.253 0.231 0.888]
    [0 0 0 0 0 1]
    [0 0 0 0 0 1]
    Nailed it
    Nailed it
    
    [5 5 4 5 6 1]
    [1 1 0 1 0 1]
    [0.776 0.738 0.175 0.744 0.179 0.883]
    [1 1 0 1 0 1]
    [1 1 0 1 0 1]
    Nailed it
    Nailed it
    
    [3 4 1 3 4 3]
    [1 0 1 1 0 1]
    [0.534 0.475 0.91  0.531 0.504 0.51 ]
    [1 0 1 1 1 1]
    [1 0 1 1 1 1]
    Nuts
    Nuts
    


There are a lot of close calls and it's not at all clear where
the ideal probability threshold is.

We can be more systematic by looking at a range of probabilities
and reporting metrics for each one.

```python
def test_threshold_precision(model, thresholds):
    """Test array of threshold values and calculate precision metrics for each.
    
    Calculate each threshold on a random sample of test data.
    Store and return in a dict.
    """
    results = dict()
    # This is going to take a while...
    size = len(X_test.values) / 10
    for threshold in thresholds:
        
        # Get sample of dice throws.
        throws = X_test.sample(size).values
        
        # Make predictions.
        y_pred = np.array([best.predict([dice])[0] for dice in throws])
        
        # Ground truth labels.
        true = np.array([make_labels(dice) for dice in throws])
        
        # Calculate metrics.
        f_one = f1_score(true, y_pred, average='samples')
        label_ranking = label_ranking_average_precision_score(true, y_pred)
        average_precision = average_precision_score(true, y_pred)
        
        # Save result.
        results[threshold] = {'f1_score': f_one,
                              'Label ranking average precision': label_ranking,
                              'Average precision': average_precision}
        
    return results
```

Start with a fairly wide range.

```python
thresholds = np.linspace(.47, .5, 10)
```


```python
threshold_test = test_threshold_precision(best, thresholds)
threshold_test
```




    {0.47: 
     {'Average precision': 0.9167155889975689,
      'Label ranking average precision': 0.972578471018679,
      'f1_score': 0.8961891757385692},
      
     0.47333333333333333: 
     {'Average precision': 0.901170639362717,
      'Label ranking average precision': 0.9731080300404387,
      'f1_score': 0.8963563588346952},
      
     0.4766666666666666: 
     {'Average precision': 0.9149462064638517,
      'Label ranking average precision': 0.9765068361255532,
      'f1_score': 0.9027246968321491},
      
     0.48: 
     {'Average precision': 0.9202440982817786,
      'Label ranking average precision': 0.9814558058925474,
      'f1_score': 0.9171755935187478},
      
     0.48333333333333334: 
     {'Average precision': 0.9261396156467017,
      'Label ranking average precision': 0.9823921625264779,
      'f1_score': 0.9266106221912115},
      
     0.48666666666666664: 
     {'Average precision': 0.9107067924289406,
      'Label ranking average precision': 0.9778572116310417,
      'f1_score': 0.9005529421473963},
      
     0.49: 
     {'Average precision': 0.9212217708645708,
      'Label ranking average precision': 0.9786828422876949,
      'f1_score': 0.9125374817749167},
      
     0.49333333333333335: 
     {'Average precision': 0.9211248772392877,
      'Label ranking average precision': 0.97738301559792,
      'f1_score': 0.9131371901735854},
      
     0.49666666666666665: 
     {'Average precision': 0.9002207841975037,
      'Label ranking average precision': 0.9749566724436739,
      'f1_score': 0.8972297873511046},
      
     0.5: 
     {'Average precision': 0.9109549311230944,
      'Label ranking average precision': 0.9731489505103024,
      'f1_score': 0.9035115952793596}}



Refine our search.

```python
thresholds = np.linspace(.476, .486, 10)
```


```python
threshold_test_1 = test_threshold_precision(best, thresholds)
threshold_test_1
```




    {0.476: 
     {'Average precision': 0.9141850012133119,
      'Label ranking average precision': 0.976015790487194,
      'f1_score': 0.905653839709299},
      
     0.4771111111111111: 
     {'Average precision': 0.9133757704663449,
      'Label ranking average precision': 0.9680290776044675,
      'f1_score': 0.9035391048389315},
      
     0.4782222222222222: 
     {'Average precision': 0.9057443026223219,
      'Label ranking average precision': 0.9741984402079722,
      'f1_score': 0.9016313168826168},
      
     0.47933333333333333: 
     {'Average precision': 0.9220965637058263,
      'Label ranking average precision': 0.9717215482380126,
      'f1_score': 0.9131454430414571},
      
     0.48044444444444445: 
     {'Average precision': 0.9118693487867908,
      'Label ranking average precision': 0.9721740804929712,
      'f1_score': 0.9045982228824516},
      
     0.4815555555555555: 
     {'Average precision': 0.9042850090379052,
      'Label ranking average precision': 0.9768727132678604,
      'f1_score': 0.9022000145050405},
      
     0.48266666666666663: 
     {'Average precision': 0.9143032678176007,
      'Label ranking average precision': 0.9743500866551126,
      'f1_score': 0.8965406728838271},
      
     0.48377777777777775: 
     {'Average precision': 0.9132657689129259,
      'Label ranking average precision': 0.9779029462738303,
      'f1_score': 0.9196355733617433},
      
     0.48488888888888887: 
     {'Average precision': 0.8875843773012423,
      'Label ranking average precision': 0.9626347968419027,
      'f1_score': 0.877161563643366},
      
     0.486: 
     {'Average precision': 0.9080659113325225,
      'Label ranking average precision': 0.9742562102830732,
      'f1_score': 0.8990564221066821}}



Refine further.

```python
thresholds = np.linspace(.482, .485, 5)
```


```python
threshold_test_2 = test_threshold_precision(best, thresholds)
threshold_test_2
```




    {0.482: 
     {'Average precision': 0.9007834319784004,
      'Label ranking average precision': 0.9668303485461194,
      'f1_score': 0.8812418565451494},
      
     0.48275: 
     {'Average precision': 0.9142902148791339,
      'Label ranking average precision': 0.9678220681686883,
      'f1_score': 0.9058636626227615},
      
     0.4835: 
     {'Average precision': 0.8963786650387666,
      'Label ranking average precision': 0.9734931638744461,
      'f1_score': 0.8870471238755467},
      
     0.48424999999999996: 
     {'Average precision': 0.9052563909226885,
      'Label ranking average precision': 0.9703061813980358,
      'f1_score': 0.8988514758878711},
      
     0.485: 
     {'Average precision': 0.9031736128238692,
      'Label ranking average precision': 0.9724845946466394,
      'f1_score': 0.9049530962009298}}



The best score so far is at a probability threshold of .48333...

```python
threshold = test_threshold_precision(best, [.48333333333333334])
threshold
```




    {0.48333333333333334: 
     {'Average precision': 0.9051624742901035,
      'Label ranking average precision': 0.9694685153090699,
      'f1_score': 0.8973054386399275}}





Average precision of 90.5% seems like a usable result 
when label ranking average precision is also very high.