from base.base_main import BaseMain
from data_gen.example_datagen import X2Data
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer


if __name__ == "__main__":
    main = BaseMain(ExampleModel, X2Data, ExampleTrainer)
    main.main(test=True)
