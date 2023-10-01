from src.data_pipeline import DataLoader
from src.model_building import ModelBuilder
from src.preprocessing import Preprocessor
from src.training import Trainer


def main():
    input_shape = (87, )
    num_classes = 9
    model_builder = ModelBuilder(input_shape, num_classes)
    model = model_builder.compile_model()
    preprocessor = Preprocessor()
    words, classes, documents = preprocessor.preprocess()
    dataloader = DataLoader(words, classes, documents)
    train_x, train_y = dataloader.create_train_data()
    trainer = Trainer(model, train_x, train_y)
    trainer.train()


if __name__ == '__main__':
    main()