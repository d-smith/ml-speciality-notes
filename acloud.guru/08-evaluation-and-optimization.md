# Evaluation and Optimization

## Intro

After training the model... is it any good?

## Evaluation and Optimization Concepts

Goal: we want generalization, not memorization.

Evaluation:

1. Define evaluation. Decide what metric or metrics we should use to decide if the algorithm is good enough.
2. Evaluate. Review the metrics during or after the training process. This might be manual or automatic, depending on the algorithm.
3. Tune. Adjust hyperparameters, data, the evaluation strategy or even the entire algoritm to bring us closer to the desired results.

Two Types:

* Offline Validation
    * Validation done using test sets of data
    * Example: validation sets and k-fold validation
* Online Validation
    * Validation under real-world conditions
    * Example: canary deployments

