from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from .collapse_time_dimension import collapse_time_dimension


def run_logistic_regression(train_data_loader, test_data_loader):
    x_train, y_train = collapse_time_dimension(train_data_loader)
    x_test, y_test = collapse_time_dimension(test_data_loader)

    clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy
