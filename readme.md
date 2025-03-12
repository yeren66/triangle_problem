# Boundary Data Experimental Results

## Experimental Setup

Building upon the previous experiment, we primarily explored the impact of boundary data on the neural network.

We sampled data from a `100*100*100` space while minimizing the training set size as much as possible. This was done both to fit real-world scenarios and to highlight the effect of boundary data.

The training set sizes were as follows:
```
Non-triangle: 2000
Scalene triangle: 5000
Isosceles triangle: 5000
Equilateral triangle: 100
```

### Definition of Boundary Data

We used Manhattan distance to measure the distance of data points from the boundary:

$d = |x_1 - x_2| + |y_1 - y_2|$

Only two classes—non-triangles and scalene triangles—were considered for boundary data. The proportion of boundary data under different thresholds was:

- When  $d < 1$ , boundary data accounted for 1.52%
- When  $d < 2$ , boundary data accounted for 10.21%
- When  $d < 3$ , boundary data accounted for 18.33%

We kept the total training set size unchanged but varied the proportion of boundary data (ranging from 0% to 100%) to analyze its impact on neural network training. For each proportion, we randomly sampled and repeated the experiment **ten times** to eliminate random errors.

## Experimental Results

- **Experiment 1: Considering boundary data where  $d < 2$**

![](fig/b_2-total-plot.png)

|  |  |
|--|--|
|![](fig/b_2-non_triangle-plot.png)|![](fig/b_2-scalene-plot.png)|
|![](fig/b_2-isoceles-plot.png)|![](fig/b_2-equilateral-plot.png)|

- **Experiment 2: Considering boundary data where  $d < 3$**

![](fig/b_3_total_plot.png)

|  |  |
|--|--|
|![](fig/b_3-non_triangle-plot.png)|![](fig/b_3-scalene-plot.png)|
|![](fig/b_3-isosceles-plot.png)|![](fig/b_3-equilateral-plot.png)|


## Visual Data Analysis

We visualized the results for ratio = 0%, 20%, 50%, 80%, and 100% as shown in the figures below.

- **ratio=0%**

![](fig/b_3-1-ratio_0.0.gif)

- **ratio=20%**

![](fig/b_3-1-ratio_0.2.gif)

- **ratio=50%**

![](fig/b_3-1-ratio_0.5.gif)

- **ratio=80%**

![](fig/b_3-1-ratio_0.8.gif)

- **ratio=100%**

![](fig/b_3-1-ratio_1.0.gif)

## Conclusion

1. As the proportion of boundary data increases, the prediction accuracy of the neural network exhibits a trend of **first increasing and then decreasing**.
2. The highest prediction accuracy is achieved when the proportion of boundary data is around **40% to 50%**.
3. Analyzing the misclassified data points reveals that when the proportion of boundary data is low, errors are primarily concentrated **near the boundary**. As the proportion of boundary data increases, these boundary errors decrease, while errors gradually shift to **non-boundary regions**.