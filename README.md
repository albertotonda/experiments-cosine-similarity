# Experiments with Cosine-Similarity kPCA 
Experiments with kPCA Cosine Similarity to explore the behavior of ensemble regressors.

## Context
We can characterize the behavior of a single weak regressor inside an ensemble as a single point in a high-dimensional space, where each dimension represents the error that that regressor makes for a specific training sample. With $d$ training samples, we have a $d$-dimensional space, that we can call *semantic space*. Unfortunately, managing and understanding high-dimensional spaces is extremely complicated: reducing the semantic space to a two-dimensional *behavior space* could make it possible to more easily interpret behavior.

## Approach
Several approaches to go from semantic space to behavior space have been tested for evolutionary ensemble machine learning techniques, and so far the most successful was kernel-principal component analysis (kPCA) with Cosine Similarity as the kernel. This series of experiments is to test whether the cosine-similarity-kPCA behavior space can also be useful to analyze the behavior of other state-of-the-art ensemble algorithms, such as catboost, lightgbm, xgboost, and the more classical Random Forest. 

## Notes
The benchmark suite selected for the project is OpenML-CTR23. Only two datasets considered in the study have missing values: fps_benchmark and Moneyball.

fps_benchmark has 3/44 columns with a lot of missing values: CpuDieSize (12960/24624, 52% missing), CpuNumberOfTransistors (12960/24624, 52% missing), GpuNumberOfComputeUnits (19152/24624, 77% missing), GpuNumberOfExecutionUnits (24624/24624, 100% missing!!!)
Moneyball has 4/15 columns with A LOT OF missing values: RankSeason (988/1232, 80% missing), RankPlayoffs (988/1232, 80% missing), OOBP (812/1232, 80% missing), OSLG (812/1232, 80% missing)

Since there are so many missing values, I think it makes sense to just ignore those columns.

## Open questions

### Algoritmo di generazione di ensemble basato su Graph Neural Networks?

### Visualize trajectory of a neural network regressor

### Visualize behavioral space for test points

### Kernel density estimation delle distribuzioni (vedi seaborn)

### Clearing the trees
Question: What happens if we just pick the trees on the edges/vertices of the convex hull in behavior space? Does the performance of the ensemble go down?
Answer: Yes, from preliminary experiments with Random Forest it looks like it does (from R2=0.56 to R2=0.51 on test). However, we go from 100 to 13 trees. Still, it's not very impressive, because taking 13 random trees in the ensemble gives a performance that is non-separable...

### Boosting in behavior space
Question: What happens if we apply Boosting, but instead of increasing the weights, we target a point in behavior space, using the inverse transformation to go back to semantic space? In other words, we are targeting a specific error value for each sample. The point to target could be the one on the other side of [0.0,0.0] in the behavior space. Still, it's a bit complicated because it does require to take into account the cumulated prediction of the trees created at every step.
Answer:

### Systematic experiments
Question: Do Random Forest's trees always cover a larger area of the cosine-similarity kPCA behavioral space? I should try on a large variety of different problems. We can check the area of the convex hull, in two dimensions it should be easy to compute.