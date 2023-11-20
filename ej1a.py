import json
import time
from functools import partial

from matplotlib import pyplot as plt

from perceptron.functions import *
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM, Momentum
from training_data.font import *
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import pandas as pd



def compare_architectures():
    architectures = config_json["architectures"]
    optimizer = ADAM
    optimizer_args = [
        config_json["optimizer"]["alpha"],
        config_json["optimizer"]["beta1"],
        config_json["optimizer"]["beta2"],
        config_json["optimizer"]["epsilon"]
    ]
    metrics = {}
    for architecture in architectures:
        architecture_name = str(architecture)
        metrics[architecture_name] = []
        for _ in range(config_json["runs"]):
            autoencoder = MultiPerceptron(
                architecture,
                partial(sigmoid, config_json["activation_function_beta"]),
                partial(sigmoid_derivative, config_json["activation_function_beta"]),
                optimizer,
                optimizer_args
            )
            error = autoencoder.train(config_json['architecture_epochs'], fonts, fonts, len(fonts))
            metrics[architecture_name].append(error)

    print(metrics)
    architectures = list(metrics.keys())
    average_errors = [round(np.mean(errors)) for errors in metrics.values()]
    std_deviation = [round(np.std(errors)) for errors in metrics.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=architectures,
        y=average_errors,
        error_y=dict(type='data', array=std_deviation, visible=True),
        marker=dict(color='lightblue'),
        textposition='outside',
        texttemplate='%{y}',
        name='Average Error'
    ))

    fig.update_layout(
        title='Average MSE per Autoencoder Architecture',
        xaxis_title='Architectures',
        yaxis_title='MSE'
    )

    fig.show()


def compare_optimizers():
    optimizers = [ADAM, Momentum]
    architecture = config_json["optimizer_architecture"]
    optimizer_args = [
        [
            config_json["optimizer"]["alpha"],
            config_json["optimizer"]["beta1"],
            config_json["optimizer"]["beta2"],
            config_json["optimizer"]["epsilon"]
        ],
        [
            config_json["momentum"]["learning_rate"],
            config_json["momentum"]["alpha"]
        ]
    ]

    metrics = {}
    for i in range(len(optimizers)):
        optimizer = optimizers[i]
        args = optimizer_args[i]

        optimizer_name = optimizer.name()
        metrics[optimizer_name] = []
        for _ in range(config_json["optimizer_runs"]):
            autoencoder = MultiPerceptron(
                architecture,
                partial(sigmoid, config_json["activation_function_beta"]),
                partial(sigmoid_derivative, config_json["activation_function_beta"]),
                optimizer,
                args
            )
            error = autoencoder.train(config_json["optimizer_limit"], fonts, fonts, len(fonts))
            metrics[optimizer_name].append(error)

    optimizers = list(metrics.keys())
    average_errors = [np.mean(errors) for errors in metrics.values()]
    std_deviation = [np.std(errors) for errors in metrics.values()]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=optimizers,
        y=average_errors,
        error_y=dict(type='data', array=std_deviation, visible=True),
        marker=dict(color='lightblue'),
        textposition='outside',
        texttemplate='%{y}',
        name='Average Error'
    ))

    fig.update_layout(
        title='Average MSE per Optimizer',
        xaxis_title='Optimizers',
        yaxis_title='MSE'
    )

    fig.show()


def mse_for_architecture(architecture, iterations):
    optimizer = ADAM
    optimizer_args = [
        config_json["optimizer"]["alpha"],
        config_json["optimizer"]["beta1"],
        config_json["optimizer"]["beta2"],
        config_json["optimizer"]["epsilon"]
    ]
    metrics = {"mse": []}
    autoencoder = MultiPerceptron(
        architecture,
        partial(sigmoid, config_json["activation_function_beta"]),
        partial(sigmoid_derivative, config_json["activation_function_beta"]),
        optimizer,
        optimizer_args
    )
    for i in range(iterations):
        error = autoencoder.train(1, fonts, fonts, len(fonts))
        metrics["mse"].append(error)

    errors = metrics.get("mse", [])
    iterations = list(range(1, len(errors) + 1))

    fig = px.scatter(x=iterations, y=errors, labels={'x': 'Iterations', 'y': 'MSE'},
                     title=f'MSE Every Iteration for Architecture {str(architecture)}')

    fig.show()

def create_new_letter(autoencoder, start_letter, end_letter, steps):

    z1 = autoencoder.encode(start_letter)
    z2 = autoencoder.encode(end_letter)

    dif = z1 - z2

    for step in range(steps + 1):
        new_letter = autoencoder.decode(z2 + (step/steps) * dif)

        # Mostramos graficos
        matrix = new_letter.reshape(7, 5)
        plt.imshow(matrix, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        plt.show()


def generate_latent_space(autoencoder):
    metrics = {}
    symbols = ["`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
               "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "DEL"]

    for i in range(len(fonts)):
        metrics[symbols[i]] = []
        latent_space = autoencoder.encode(fonts[i])
        print(latent_space)
        metrics[symbols[i]] = latent_space.tolist()

    # Extract keys and coordinates
    keys = list(metrics.keys())
    coordinates = list(metrics.values())

    # Create a scatter plot
    plt.scatter(*zip(*coordinates), marker='o')

    # Annotate each point with its key
    for i, key in enumerate(keys):
        plt.annotate(key, (coordinates[i][0], coordinates[i][1]), textcoords="offset points", xytext=(0, 5),
                     ha='center')

    # Set the title
    plt.title('Latent Space')

    # Show the plot
    plt.show()


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
        optimizer_args = [config_json["momentum"]["learning_rate"],
                          config_json["momentum"]["alpha"]]

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

    if config_json["test"]:
        print("\n- = - = - RUNNING TESTS - = - = -\n")
        autoencoder.test(fonts, fonts)

    generate_latent_space(autoencoder)

    if config_json["compare_architectures"]:
        compare_architectures()

    if config_json["compare_optimizers"]:
        compare_optimizers()

    if config_json["mse_for_architecture"]:
        mse_for_architecture(config_json["architecture"], config_json["mse_iterations"])

    if config_json["create_new_letter"]:
        create_new_letter(autoencoder, fonts[config_json["start_letter_idx"]], fonts[config_json["end_letter_idx"]], config_json["steps"])
