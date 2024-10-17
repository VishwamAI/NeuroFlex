class AdaptiveLearningRateScheduler:
    def __init__(self, initial_lr=0.001, patience=10, factor=0.5):
        self.lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.best_performance = float('-inf')
        self.wait = 0

    def step(self, performance):
        if performance > self.best_performance:
            self.best_performance = performance
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.lr *= self.factor
                self.wait = 0
        return self.lr

def create_adaptive_learning_rate_scheduler(initial_lr=0.001, patience=10, factor=0.5):
    return AdaptiveLearningRateScheduler(initial_lr, patience, factor)
