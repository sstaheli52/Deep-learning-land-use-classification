# Early stopping class
class EarlyStopper:
    def __init__(self, patience=2, min_delta=0.001):
        self.patience = patience      # epochs to wait after last improvement
        self.min_delta = min_delta    # minimum change to count as improvement
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Save a copy of the best weights
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience  # True = stop training

    def restore_best_weights(self, model):
        if self.best_weights:
            model.load_state_dict(self.best_weights)