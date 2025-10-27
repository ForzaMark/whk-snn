from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from util.create_data_loader import create_data_loader



CLASSIFIERS = [
    (
        "logistic regression",
        LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000),
    ),
    ("svm", SVC()),
]

if __name__ == "__main__":
    train_data_loader, test_data_loader = create_data_loader()

    x_train, y_train = collapse_time_dimension(train_data_loader)
    x_test, y_test = collapse_time_dimension(test_data_loader)

    for name, clf in CLASSIFIERS:
        print(f"### classification {name} ###")
        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {accuracy:.2f}")
