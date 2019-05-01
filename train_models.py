import numpy as np
import data_extraction
from sklearn import model_selection
from sklearn import neighbors
from sklearn import tree


# choose code to run at the bottom of the file
# this file should be located in the same folder as the one A2_Data
# is located in. A2_Data should be structured the same way as it was when
# uploaded to piazza or as shown in readme.txt
def step_3_classifier():
    pocket_x, pocket_y, activity_list = (
        data_extraction.step_2_feature_extraction('./A2_Data/Pocket/')
    )
    hand_x, hand_y, activity_list = data_extraction.step_2_feature_extraction(
        './A2_Data/Hand/'
    )
    X = np.concatenate((pocket_x, hand_x))
    y = np.concatenate((pocket_y, hand_y))
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=50, stratify=y
    )

    train_KNN(X_train, X_test, y_train, y_test, activity_list)

    train_DT(X_train, X_test, y_train, y_test, activity_list)

def step_4_classifier():
    pocket_x, pocket_y, activity_list = (
        data_extraction.step_2_feature_extraction('./A2_Data/Pocket/')
    )
    hand_x, hand_y, activity_list = data_extraction.step_2_feature_extraction(
        './A2_Data/Hand/'
    )
    print('step 4a')
    train_KNN(pocket_x, hand_x, pocket_y, hand_y, activity_list)
    train_DT(pocket_x, hand_x, pocket_y, hand_y, activity_list)
    print('step 4b')
    train_KNN(hand_x, pocket_x, hand_y, pocket_y, activity_list)
    train_DT(hand_x, pocket_x, hand_y, pocket_y, activity_list)

def step_5_classifier():
    pocket_x, pocket_y, activity_list = (
        data_extraction.step_2_feature_extraction('./A2_Data/Pocket/')
    )
    hand_x, hand_y, activity_list = data_extraction.step_2_feature_extraction(
        './A2_Data/Hand/'
    )
    X = np.concatenate((pocket_x, hand_x))
    y = np.concatenate((pocket_y, hand_y))
    
    print('time domain only')
    # time domain features only
    X_TD = np.concatenate((X[:, :6], X[:,9:15]), axis=1)
    X_train_TD, X_test_TD, y_train, y_test = model_selection.train_test_split(
        X_TD, y, test_size=0.2, random_state=50, stratify=y
    )
    train_KNN(X_train_TD, X_test_TD, y_train, y_test, activity_list)
    train_DT(X_train_TD, X_test_TD, y_train, y_test, activity_list)
    print('\n')
    print('frequency domain only')
    # frequency domain features only
    X_FD = np.concatenate((X[:,6:9], X[:,15:18]), axis=1)
    X_train_FD, X_test_FD, y_train, y_test = model_selection.train_test_split(
        X_FD, y, test_size=0.2, random_state=50, stratify=y
    )
    train_KNN(X_train_FD, X_test_FD, y_train, y_test, activity_list)
    train_DT(X_train_FD, X_test_FD, y_train, y_test, activity_list)


           
###############################################################################

# choose the best KNN classifier and predict with it, then evaluate it based
# on the test data in each activity individually
def train_KNN(X_train, X_test, y_train, y_test, activity_list):
    KNNclassifier = neighbors.KNeighborsClassifier()
    n_neighbor_amounts = range(1,55,5)
    print('start KNN cross validation')
    best_n_neighbors, best_score = pick_KNN(
        KNNclassifier, n_neighbor_amounts, X_train, y_train
    )
    print('best number of neighbors ', best_n_neighbors)
    print('best total score of 5 fold cv: ', best_score)
    # do it again with smaller intervals
    n_neighbor_amounts = range(max(1, best_n_neighbors-5), best_n_neighbors+5)
    best_n_neighbors, best_score = pick_KNN(
        KNNclassifier, n_neighbor_amounts, X_train, y_train
    )
    print('best number of neighbors ', best_n_neighbors)
    print('best total score ', best_score)

    KNNclassifier.n_neighbors = best_n_neighbors
    KNNclassifier.fit(X_train, y_train)
    predictions = KNNclassifier.predict(X_test)
    # get the precision recall and accuracy for each activity
    print('KNN classifier')
    evaluate_predictions(predictions, y_test, activity_list)

# same as train_KNN but with decision tree
def train_DT(X_train, X_test, y_train, y_test, activity_list):
    DTclassifier = tree.DecisionTreeClassifier()
    max_depth_amounts = range(3,35,5)
    print('start decision tree classifier')
    best_max_depth, best_score = pick_DT(
        DTclassifier, max_depth_amounts, X_train, y_train
    )
    print('best max_depth ', best_max_depth)
    print('best total score of 5 fold cv: ', best_score)

    max_depth_amounts = range(max(best_max_depth-5, 2), best_max_depth +5)
    best_max_depth, best_score = pick_DT(
        DTclassifier, max_depth_amounts, X_train, y_train
    )
    print('best max_depth ', best_max_depth)
    print('best total score of 5 fold cv: ', best_score)
    DTclassifier.max_depth = best_max_depth
    DTclassifier.fit(X_train, y_train)
    predictions = DTclassifier.predict(X_test)
    # get the precision recall and accuracy for each activity
    print('DT classifier')
    evaluate_predictions(predictions, y_test, activity_list)

# given some predictions and the true values, prints the recall, 
# precision and accuracy on each activity
def evaluate_predictions(predictions, y_test, activity_list):
    for i, activity in enumerate(activity_list):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        true_neg = 0
        for j, prediction in enumerate(predictions):
            if prediction == i:
                if prediction != y_test[j]:
                    false_pos = false_pos + 1
                else:
                    true_pos = true_pos + 1
            elif y_test[j] == i:
                false_neg = false_neg + 1
            else:
                true_neg = true_neg + 1
        print(activity)
        print(
            'recall ', true_pos/(true_pos+false_neg),
            '\nprecision ', true_pos/(true_pos + false_pos),
            '\naccuracy ', (true_neg + true_pos)/predictions.shape[0]
        )

# pick the best KNN model from models with different numbers of neighbors
def pick_KNN(KNNclassifier, n_neighbor_amounts, X_train, y_train):
    best_n_neighbors = 0
    best_score = 0
    for n_neighbors in n_neighbor_amounts:
        KNNclassifier.n_neighbors=n_neighbors
        cv_scores = model_selection.cross_val_score(
            KNNclassifier, X_train, y=y_train, cv=5
        )
        print(cv_scores)
        score = np.sum(cv_scores)
        if(score > best_score):
            best_n_neighbors = n_neighbors
            best_score = score
    return best_n_neighbors, best_score

# same as pick_KNN but with decision tree and different maximum depths
def pick_DT(DTclassifier, max_depth_amounts, X_train, y_train):
    best_score = 0
    best_max_depth = 0
    for max_depth in max_depth_amounts:
        DTclassifier.max_depth = max_depth
        cv_scores = model_selection.cross_val_score(
            DTclassifier, X_train, y_train, cv=5
        )
        print(cv_scores)
        total_score = np.sum(cv_scores)
        if total_score > best_score:
            best_score = total_score
            best_max_depth = max_depth
    return best_max_depth, best_score
###############################################################################
# choose which steps to run here



print('start step 3')
step_3_classifier()
print('start step 4')
step_4_classifier()
print('start step 5')
step_5_classifier()