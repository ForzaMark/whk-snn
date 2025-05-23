class EarlyStopping:
    def __init__(self, patience=5, min_delta: float=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, test_accuracy):
        if self.best_score is None:
            self.best_score = test_accuracy
        elif test_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = test_accuracy
            self.counter = 0
