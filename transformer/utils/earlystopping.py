class EarlyStopping:

    def __init__(
            self,
            delta: float = 0.1,
            patience: int = 20):

        self.delta = delta
        self.patience = patience

        self.counter: int = 0
        self.smallest_loss: float = float("inf")

        self.should_save: bool = False
        self.should_stop: bool = False

    def step(self, val_loss: float) -> None:

        if self.smallest_loss is float("inf"):
            self.smallest_loss = val_loss

            self.should_save = True
            self.should_stop = False

        elif val_loss > self.smallest_loss + self.delta:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_save = False
                self.should_stop = True

        else:
            self.smallest_loss = val_loss
            self.counter = 0

            self.should_save: bool = True
            self.should_stop: bool = False
