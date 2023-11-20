import json
import random
from functools import partial
import numpy as np
from perceptron.functions import sigmoid, sigmoid_derivative
from perceptron.optimizers import ADAM, Momentum
from perceptron.vae import VAE
from training_data.emoji import emojis
import plotly.express as px


def print_emoji(array):
    i = 0
    for elem in array:
        i += 1
        if elem < 0.5:
            print(0, end="")
        else:
            print(1, end="")
        if i % 22 == 0:
            print()


if __name__ == "__main__":

    with open("ej2_config.json", "r") as f:
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

    emojis = [np.array(elem) for elem in emojis]

    vae = VAE(
        config_json["layer_config"],
        partial(sigmoid, config_json["activation_function_beta"]),
        partial(sigmoid_derivative, config_json["activation_function_beta"]),
        optimizer,
        optimizer_args
    )

    vae.train(config_json["epochs"], emojis)

    z_space = []
    for emoji in emojis:
        z_space.append(vae.encode(emoji))

    x_values, y_values = zip(*z_space)

    fig = px.scatter(x=x_values, y=y_values, labels={'x': 'X-axis', 'y': 'Y-axis'}, title='2D Scatter Plot')

    fig.show()

    # z1 = vae.encode(emojis[0])
    # z2 = vae.encode(emojis[1])
    #
    # dif = z1 - z2
    # total_steps = 5
    #
    # for step in range(total_steps):
    #     new = z2 + (step/total_steps) * dif
    #
    #     e = vae.decode(new)
    #     print_emoji(e)
    #     print("\n-----------------------------------\n")





