import numpy as np
import tensorflow.keras as tfk


class Trainer:
    """Trains model.

        Attributes:
            - model
            - train_x
            - train_y
            - epochs
            - batch_size
        
        Method:
            - train
    """
    def __init__(self, model: tfk.Model, train_x: np.array, train_y: np.array,
                 epochs: int = 2, batch_size: int = 5) -> None:
        self.model = model
        self.train_x = train_x
        self.train_y = train_y
        self.epochs = epochs
        self.batch_size = batch_size

    def train(self):
        """Fits the model and save the weights.
        """
        history = self.model.fit(self.train_x, self.train_y, epochs = self.epochs,
                                 batch_size = self.batch_size, verbose = 1)
        self.model.save_weights('chatbot.h5', history)
