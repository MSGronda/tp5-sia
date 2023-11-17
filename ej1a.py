import json
import time
from functools import partial
from perceptron.functions import *
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM, Momentum
from training_data.font import *


if __name__ == "__main__":
    with open("ej1a_config.json", "r") as f:
        config_json = json.load(f)

    if config_json["seed"] != -1:
        random.seed(config_json["seed"])
        np.random.seed(config_json["seed"])

    if config_json["optimizer"]["type"] == "adam":
        optimizer = ADAM
        optimizer_args = [
            config_json["optimizer"]["alpha"],
            config_json["optimizer"]["beta1"],
            config_json["optimizer"]["beta2"],
            config_json["optimizer"]["epsilon"]
        ]

    elif config_json["optimizer"]["type"] == "momentum":
        optimizer = Momentum
        optimizer_args = [config_json["optimizer"]["learning_rate"]]

    else:
        RuntimeError("Invalid optimizer")
        quit()

    autoencoder = MultiPerceptron(
        config_json["layer_config"],
        partial(sigmoid, config_json["activation_function_beta"]),
        partial(sigmoid_derivative, config_json["activation_function_beta"]),
        optimizer,
        optimizer_args
    )

    batch_size = len(fonts) if config_json["batch_size"] == -1 else config_json["batch_size"]

    t1 = time.time()
    min_error = autoencoder.train(
        config_json["epochs"],
        fonts,
        fonts,
        batch_size
    )
    t2 = time.time()

    print("\n- = - = - FINISHED - = - = -\n")
    print(f"Min error (MSE): {min_error}")
    print(f"Time taken: {t2-t1} s")
    print("\n- = - = - RUNNING TESTS - = - = -\n")

    autoencoder.test(fonts, fonts)
