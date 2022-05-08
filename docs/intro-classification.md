(section-cl-intro)=
# Basics

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
