import pandas as pd
import matplotlib.pyplot as plt
from surprise import Dataset, SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF


from surprise.model_selection import cross_validate, KFold

print('Imports successful!')

data = Dataset.load_builtin('ml-100k')
print('\n\nData load successful!')

# 1. Get the ratings file from the data object
# This is just a filename that has all the data stored in it
ratings_file = data.ratings_file

# 2. Load that table using pandas, a commmon python data loading tool
# We set the column names manually here
col_names = ['user_id', 'item_id', 'rating', 'timestamp']
raw_data = pd.read_table(ratings_file, names=col_names)

# 3. Get the rating column
ratings = raw_data.rating

# 4. Generate a bar plot/histogram of that data
ratings.value_counts().sort_index().plot.bar()

plt.show()
print('\n\nHistogram generation successful!')

# Create model object
model_random = NormalPredictor()
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
model_random_results = cross_validate(model_random, data, measures=['RMSE'], cv=5, verbose=True)
print('\n\nModel training successful!')

# Create model object
model_user = KNNBasic(sim_options={'user_based': True})
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may have a lot of print output
# You can set verbose=False to prevent this from happening
model_user_results = cross_validate(model_user, data, measures=['RMSE'], cv=5, verbose=True)
print('\n\nModel training successful!')

# Create model object
model_item = KNNBasic(sim_options={'user_based': False})
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may have a lot of print output
# You can set verbose=False to prevent this from happening
model_item_results = cross_validate(model_item, data, measures=['RMSE'], cv=5, verbose=True)
print('\n\nModel training successful!')


# Create model object
model_matrix = SVD()
print('Model creation successful!')


# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may take some time (2-3 minutes) to train, so please be patient
model_matrix_results = cross_validate(model_matrix, data, measures=['RMSE'], cv=5, verbose=True)
print('\n\nModel training successful!')

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = dict()
    for uid, _, true_r, est, _ in predictions:
        current = user_est_true.get(uid, list())
        current.append((est, true_r))
        user_est_true[uid] = current

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

print('Function creation successful!')

#The next function returns the name of the model
def  modelName(model):
    if model == model_random:
        return 'random'
    elif model == model_item:
        return 'item-item'
    elif model == model_user:
        return 'user-user'
    elif model == model_matrix:
        return 'matrix'
#definition of the folds and the cycle that is going to compute the precisions al recalls
kf = KFold(n_splits=5)
lst = [5, 10]
mls = [model_random, model_user, model_item, model_matrix]
for i in mls:
    for j in lst:
        sumPre = 0
        sumRec = 0
        for trainset, testset in kf.split(data):
            i.fit(trainset)
            predictions = i.test(testset)
            precisions, recalls = precision_recall_at_k(predictions, k=j, threshold=3.5)
            sumPre = sumPre + sum(pr for pr in precisions.values()) / len(precisions)
            sumRec = sumRec + sum(rec for rec in recalls.values()) / len(recalls)
            # Precision and recall can then be averaged over all users
        meanPre = sumPre/5
        meanRec = sumRec/5
        print('The  average precision for ',  modelName(i),  'with k=' + str(j) + ' is ', meanPre)
        print('The  average recall for ', modelName(i),  'with k=' + str(j) + ' is ', meanRec)

def get_top_n(predictions, n=5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = dict()
    for uid, iid, true_r, est, _ in predictions:
        current = top_n.get(uid, [])
        current.append((iid, est))
        top_n[uid] = current

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

print('Function creation successful!')

trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
print('Trainset and testset creation successful!')


for i, model in enumerate(mls):
    model.fit(trainset)
    predictions = model.test(testset)
    top_n = get_top_n(predictions, n=5)
    # Print the first one
    user = list(top_n.keys())[0]
    print(f'model name: {modelName(i)}')
    print(f'user ID: {user}')
    print(f'top 5 movie ID\'s this user would like, sorted by rating highest to lowest: {top_n[user]}')

print('\n\nTop N computation successful! YOU ARE DONE WITH THE CODE!')