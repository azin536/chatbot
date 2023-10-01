import tensorflow.keras as tfk


class ModelBuilder:
    """Builds and compile model.

       Attributes:
        - input_shape
        - num_classes
     
       Methods:
        - _build model
        - compile model
    """
    def __init__(self, input_shape: int, num_classes: int) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self) -> tfk.Model:
        """Builds the model

        Args:
            input_shape (int)
            num_classes (int)

        Returns:
            tfk.Model: built model
        """
        model = tfk.models.Sequential()
        model.add(tfk.layers.Dense(128, input_shape=self.input_shape, activation='relu'))
        model.add(tfk.layers.Dropout(0.5))
        model.add(tfk.layers.Dense(64, activation='relu'))
        model.add(tfk.layers.Dropout(0.5))
        model.add(tfk.layers.Dense(self.num_classes, activation='softmax'))
        return model

    def compile_model(self):
        model = self.build_model()
        sgd = tfk.optimizers.legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

