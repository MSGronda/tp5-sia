import time
import json
from functools import partial
from perceptron.functions import *
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM, Momentum
from training_data.font import *

if __name__ == "__main__":

    with open("ej1b_config.json", "r") as f:
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
        quit("Invalid optimizer")

    autoencoder = MultiPerceptron(
        config_json["layer_config"],
        partial(sigmoid, config_json["activation_function_beta"]),
        partial(sigmoid_derivative, config_json["activation_function_beta"]),
        optimizer,
        optimizer_args
    )

    t1 = time.time()
    min_error = autoencoder.train(
        config_json["epochs"],
        fonts,
        fonts,
        config_json["batch_size"]
    )
    t2 = time.time()
    print(min_error, t2 - t1)

    if config_json["strategy"] == "bit_flip_with_probability":
        flip_probability_min = config_json["probability_min"]
        flip_probability_max = config_json["probability_max"]
        step = config_json["step"]

        with open("results/results_denoising.csv", "w") as f:
            current_prob = flip_probability_min
            while current_prob <= flip_probability_max:

                noised_fonts = copy.deepcopy(fonts)

                for vec in noised_fonts:
                    bit_fliping_with_probability(vec, current_prob)

                counter_match = autoencoder.test(noised_fonts, fonts)
                print(f"{current_prob},{counter_match}", file=f)

                current_prob += step
    elif config_json["strategy"] == "bit_fliping_with_n":

        min_n = config_json["n_min"]
        max_n = config_json["n_max"]
        if max_n > 30:
            quit("Max n to high, max 30")

        with open("results/results_denoising_n.csv", "w") as f:

            current_n = min_n
            while current_n <= max_n:

                noised_fonts = copy.deepcopy(fonts)

                for vec in noised_fonts:
                    bit_fliping_with_n(vec, current_n)

                counter_match = autoencoder.test(noised_fonts, fonts)
                print(f"{current_n},{counter_match}", file=f)

                current_n += 1
    else:
        quit("Invalid noising strategy")
