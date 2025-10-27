from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from .collapse_time_dimension import collapse_time_dimension


def run_svm(train_data_loader, test_data_loader):
    x_train, y_train = collapse_time_dimension(train_data_loader)
    x_test, y_test = collapse_time_dimension(test_data_loader)

    clf = SVC()

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
