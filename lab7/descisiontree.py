import pandas as pd
import numpy as np

df = pd.read_csv('loan_approval_dataset.csv')
df.columns = df.columns.str.strip()
print(df.columns)
df.drop("loan_id", axis=1, inplace=True)
df["education"] = df["education"].str.strip().map({
    "Graduate": 1,
    "Not Graduate": 0
})
df["self_employed"] = df["self_employed"].str.strip().map({
    "Yes": 1,
    "No": 0
})
df["loan_status"] = df["loan_status"].str.strip().map({
    "Approved": 1,
    "Rejected": 0
})

print(df.head())

features = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]
X = df[features]
y = df["loan_status"]

def custom_train_test_split(X, y, test_size=0.2):
    shuffled_indices = np.random.permutation(len(X))
    test_count = int(len(X) * test_size)
    test_indices = shuffled_indices[:test_count]
    train_indices = shuffled_indices[test_count:]
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = custom_train_test_split(X, y)

def gini_impurity(target_values):
    unique_classes, class_counts = np.unique(target_values, return_counts=True)
    gini_score = 1
    for class_count in class_counts:
        class_probability = class_count / len(target_values)
        gini_score -= class_probability ** 2
    return gini_score

print("Gini Impurity:", gini_impurity(y))
print(X['loan_amount'])

def best_split(X, y):
    best_feature = None
    best_threshold = None
    lowest_gini = float("inf")
    for feature_name in X.columns:
        possible_thresholds = X[feature_name].unique()
        for threshold in possible_thresholds:
            left_side_mask = X[feature_name] <= threshold
            right_side_mask = X[feature_name] > threshold
            left_side_targets = y[left_side_mask]
            right_side_targets = y[right_side_mask]
            if len(left_side_targets) == 0 or len(right_side_targets) == 0:
                continue
            left_side_gini = gini_impurity(left_side_targets)
            right_side_gini = gini_impurity(right_side_targets)
            weighted_gini = (
                (len(left_side_targets) / len(y)) * left_side_gini
                + (len(right_side_targets) / len(y)) * right_side_gini
            )
            if weighted_gini < lowest_gini:
                lowest_gini = weighted_gini
                best_feature = feature_name
                best_threshold = threshold
    return best_feature, best_threshold, lowest_gini

feature, threshold, gini = best_split(X_train, y_train)
print("Best Feature:", feature)
print("Best Threshold:", threshold)
print("Best Gini:", gini)

def majority_class(y):
    # Count each class and return the one with highest count
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

def build_tree(X, y, depth=0, max_depth=4):
    # Stop if all output labels are same (pure node)
    if len(np.unique(y)) == 1:
        return {"value": y.iloc[0]}
    # Stop if we reached the allowed tree depth
    if depth == max_depth:
        return {"value": majority_class(y)}
    # Find best column and value to split on
    feature, threshold, _ = best_split(X, y)
    # If no good split exists, return majority class
    if feature is None:
        return {"value": majority_class(y)}
    # Split rows into left (<= threshold) and right (> threshold)
    left_mask = X[feature] <= threshold
    right_mask = ~left_mask
    # Recursively build left and right subtrees
    return {
        "feature": feature,
        "threshold": threshold,
        "left": build_tree(X[left_mask], y[left_mask], depth + 1, max_depth),
        "right": build_tree(X[right_mask], y[right_mask], depth + 1, max_depth)
    }

def predict_one(tree, row):
    # Start from root and move down until leaf is reached
    current = tree
    while "value" not in current:
        feature = current["feature"]
        threshold = current["threshold"]
        # Go left for <= threshold, else go right
        if row[feature] <= threshold:
            current = current["left"]
        else:
            current = current["right"]
    # Leaf node stores final class
    return current["value"]

def predict_tree(tree, X):
    # Predict class for each row in X
    # X.iloc[i] returns the i-th row of the DataFrame X as a Series object.
    # In this context, it is used to select each sample (row) from X,
    # so that predict_one(tree, X.iloc[i]) can make a prediction using the tree for that specific sample.
    predictions = [predict_one(tree, X.iloc[i]) for i in range(len(X))]
    return np.array(predictions)

# Qn–3: Prediction
# Predict loan approval using Decision Tree
# Output class label (0 or 1)
tree = build_tree(X_train, y_train)
y_pred = predict_tree(tree, X_test)
print("Predicted class labels (0/1):", y_pred)

# Qn–4: Random Forest Implementation
def bootstrap_sample(X, y):
    # Sample row indices with replacement
    sample_indices = np.random.choice(len(X), size=len(X), replace=True)
    return X.iloc[sample_indices], y.iloc[sample_indices]

def train_random_forest(X, y, n_trees):
    # Train multiple trees on different bootstrap samples
    forest = []
    for _ in range(n_trees):
        X_sample, y_sample = bootstrap_sample(X, y)
        tree = build_tree(X_sample, y_sample)
        forest.append(tree)
    return forest

def majority_vote(predictions):
    # predictions shape: (n_trees, n_samples)
    final_predictions = []
    for sample_preds in predictions.T:
        values, counts = np.unique(sample_preds, return_counts=True)
        final_predictions.append(values[np.argmax(counts)])
    return np.array(final_predictions)

def predict_forest(forest, X):
    # Collect predictions from all trees, then vote
    all_tree_predictions = np.array([predict_tree(tree, X) for tree in forest])
    return majority_vote(all_tree_predictions)

forest = train_random_forest(X_train, y_train, n_trees=5)
forest_predictions = predict_forest(forest, X_test)
print("Random Forest predicted labels (0/1):", forest_predictions)