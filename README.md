# Experiments with Cosine-Similarity kPCA 
Experiments with kPCA Cosine Similarity to explore the behavior of ensemble regressors.

## Context
We can characterize the behavior of a single weak regressor inside an ensemble as a single point in a high-dimensional space, where each dimension represents the error that that regressor makes for a specific training sample. With $d$ training samples, we have a $d$-dimensional space, that we can call *semantic space*. Unfortunately, managing and understanding high-dimensional spaces is extremely complicated: reducing the semantic space to a two-dimensional *behavior space* could make it possible to more easily interpret behavior.

## Approach
Several approaches to go from semantic space to behavior space have been tested for evolutionary ensemble machine learning techniques, and so far the most successful was kernel-principal component analysis (kPCA) with Cosine Similarity as the kernel. This series of experiments is to test whether the cosine-similarity-kPCA behavior space can also be useful to analyze the behavior of other state-of-the-art ensemble algorithms, such as catboost, lightgbm, xgboost, and the more classical Random Forest. 
