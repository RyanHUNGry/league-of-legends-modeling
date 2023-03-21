# League of Legends Exploratory Data Analysis
### Creator: Ryan Hung
---
## Framing the Problem:
### Problem:

This modeling notebook employs **[Oracle Elixir's](https://oracleselixir.com/tools/downloads)** dataset consisting of player and team data from the 2022 League of Legends competitive season.

The data used here has already been cleaned in the exploratory data analysis found **[here](https://ryanhungry.github.io/league-of-legends-EDA/)**.

League of Legends is a highly complex game, and victory is typically determined through various factors. 

This modeling notebook focuses on creating a binary classifier to predict whether or not a team will win or lose a game. We will be using the response variable `result`, which is a boolean column where `True` represents a win and `False` represents a loss. We are using `result` as our response variable because it directly relates to our problem of predicting whether or not a team will win.

We will be utilizing accuracy over other metrics such as precision, recall, or F1 score because predicting a false positive or a false negative is equally bad in our case. Moreover, there is no class imbalance in our data. In other words, the proportion of teams that win is roughly equal to the proportion of teams that lose in our dataset. To better visualize this, we provide a plot of the class distribution in our training set. 

Lastly, we need to specify what features are available at the time of prediction. In our model, we will be using the features `firstbaron`, `firstblood`, and other features captured at the 15-minute mark of a match. All of these features are available during the time of prediction because we have access to them before the game has finished. In other words, there is no information that is only available after the game has ended.

---
## Baseline Model:
### Baseline Model:
The model we are using is logistic regression because it is best used for binary classification problems. For the baseline model, we are only using the quantitative discrete features `firstbaron` and `firstblood`. These two features are boolean True/False features, so we can use one-hot encoding to represent them as numerical 0/1s. To do so, we used `Onehotencoder`, put it inside a `ColumnTransformer`, listed the columns `firstblood` and `firstbaron`, and then bundled everything into a `Pipeline` with our `LogisticRegression` model.

We chose these two features for our baseline because getting the first baron or the first blood benefits a team, and so there is an intuitive positive correlation between getting these objectives and winning.

After fitting our model on the training set and then evaluating accuracy on the test set, we achieve a baseline accuracy of **0.8199**. This performance is good as a baseline given we are only using two features. We achieved this decent baseline performance because our assumption of first baron and first blood benefitting team is correct. 

---
## Final Model:
### Final Model:
For our final model, we are adding two additional features to our original baseline model:
1. We engineer the feature `kda_diff_at_15`, which is the difference between 
