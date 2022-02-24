(section-cl-intro)=
# Introduction

In supervised learning, we have a dataset consisting of both features (usually multiple `X` variables) and labels (our `y` variable). The task is to construct an estimator (model) which is able to predict the label `y` of an object given the set of features `X`. 

Supervised learning is further broken down into two categories: 

- classification
- regression

In classification, the label is discrete, while in regression, the label is continuous.

:::{note}
Classification refers to a predictive modeling problem where a categorical class label is predicted.
:::

Examples of classification problems include:

- Given an example, classify if it is spam or not.
- Given a handwritten character, classify it as one of the known characters.
- Given recent user behavior, classify as churn or not.

In the next sections, we'll cover the primary building blocks of classification models. Therefore, we mainly use content from Google's excellent Machine Learning Crash Course.

## Confusion matrix

We'll start with the metrics we'll use to evaluate classification models: 

<br>

<iframe src="https://docs.google.com/presentation/d/e/2PACX-1vSzKkW3nZkCQPXapPzHFsoVdH0_sIsxz1psMgeqPb0Gg-AVolox0R06dFqSfEVj8tIdqm5mIxsg85zG/embed?start=false&loop=false&delayms=3000" frameborder="0" width="820" height="608" allowfullscreen="true" mozallowfullscreen="true" webkitallowfullscreen="true"></iframe>

<br>

```{admonition} Resources
:class: tip

- [Download slides](https://docs.google.com/presentation/d/1Uait12xIaTM9rdyxSEnzIbR8opENnoOOPefuPocze3s/export/pdf)

- Reading: (Classification: True vs. False and Positive vs. Negative)[https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative?hl=en]

```

## Performance metrics

### Accuracy

Accuracy is one metric for evaluating classification models:

- Read: [Accuracy](https://developers.google.com/machine-learning/crash-course/classification/accuracy?hl=en)

### Precision and recall


Learn about precision and recall:

- Read:[Precision and Recall](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall?hl=en)

Next, check your understanding by answering [this questions about Accuracy, Precision and Recall)](https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-accuracy-precision-recall?hl=en)

Read the interactive article [Attacking discrimination with smarter machine learning](https://research.google.com/bigpicture/attacking-discrimination-in-ml/) to get a better understanding of the relevance of thresholds in classification problems.

### ROC and AUC

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds.

- Read: [ROC Curve and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=en)

Check your understanding: [ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/check-your-understanding-roc-and-auc?hl=en)


## Unbalanced data

Finally, we'll cover a common problem in machine learning tasks: unbalanced data.

```{admonition} Resources
:class: tip
- [Download slides](https://drive.google.com/file/d/1LQqDr_ykos1Aw9Ht80pnbSed_8KSmRMO/view?usp=sharing)
```