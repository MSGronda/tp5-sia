import json
import random
from functools import partial
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from perceptron.functions import sigmoid, sigmoid_derivative
from perceptron.optimizers import ADAM, Momentum
from perceptron.vae import VAE
from training_data.emoji import emojis
import plotly.express as px
import plotly.graph_objects as go


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


def graph_latent_space(vae, emojis):
    z_space = []
    labels = ["🙂", "☹️", "😥", "😂", "😜", "😋", "😝", "😮", "😖"]

    for emoji in emojis:
        z_space.append(vae.encode(emoji))

    x_values, y_values = zip(*z_space)

    fig = px.scatter(x=x_values, y=y_values, text=labels, labels={'x': 'Mu', 'y': 'Sigma'}, title='2D Scatter Plot')

    fig.update_layout(annotations=[
        dict(x=x_val, y=y_val, text=label, showarrow=False)
        for x_val, y_val, label in zip(x_values, y_values, labels)
    ])

    fig.show()

def draw_emoji(emoji):
    matrix = emoji.reshape(25, 22)
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.show()


def draw_outputs(vae, emojis):
    for emoji in emojis:
        draw_emoji(vae.forward_propagation(emoji))

def graph_loss(vae):

    x_values = [i for i in range(len(vae.r_loss))]
    y_values_list1 = vae.r_loss
    y_values_list2 = vae.kl_loss
    y_values_list3 = vae.total_loss

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_values, y=y_values_list1, mode='markers', name='Reconstruction Loss', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_values, y=y_values_list2, mode='markers', name='KL Loss', marker=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_values, y=y_values_list3, mode='markers', name='Total Loss', marker=dict(color='blue')))

    fig.update_layout(title='Loss vs Epochs',
                      xaxis=dict(title='Epoch'),
                      yaxis=dict(title='Loss'))
    fig.show()

def save_emoji(emoji, i, j):
    matrix = emoji.reshape(25, 22)
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.savefig(f"./results/images/{i}-{j}.png", bbox_inches='tight', pad_inches=0)
    plt.clf()

def add_noise(emoji):
    r = np.random.uniform(0, 0.15, size=emoji.shape)
    for i in range(len(emoji)):
        if emoji[i] > 0.15:
            emoji[i] += r[i]

    return emoji

def generate_new_emoji(vae, emojis, total_steps):

    mean = []
    std = []
    for emoji in emojis:
        m, s = vae.encode(emoji)
        mean.append(m)
        std.append(s)

    mean = np.array(mean)
    std = np.array(std)

    min_mean = np.min(mean)
    max_mean = np.max(mean)

    min_std = np.min(std)
    max_std = np.max(std)

    mean_values = [min_mean + (step / total_steps) * (max_mean - min_mean) for step in range(total_steps + 1)]
    std_values = [min_std + (step / total_steps) * (max_std - min_std) for step in range(total_steps + 1)]


    for i, m in enumerate(mean_values):
        for j, s in enumerate(std_values):
            new_emoji = vae.decode(np.array([m,s]))
            save_emoji(new_emoji, i, j)

    # Define the number of rows and columns in the grid
    num_rows, num_cols = total_steps + 1, total_steps + 1

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(2*num_rows, 2*num_cols))

    # Loop through each position in the grid
    for i in range(num_rows):
        for j in range(num_cols):
            # Construct the filename based on i and j values
            filename = f"./results/images/{i}-{j}.png"

            # Load the image using matplotlib.image
            img = mpimg.imread(filename)

            # Display the image in the corresponding subplot
            axes[i, j].imshow(img)

            # Hide axes ticks and labels for better visualization
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the grid of images
    plt.show()



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

    if config_json["graph_latent_space"]:
        graph_latent_space(vae, emojis)

    if config_json["draw_outputs"]:
        draw_outputs(vae, emojis)

    if config_json["graph_loss"]:
        graph_loss(vae)

    if config_json["generate_new_emoji"]:
        generate_new_emoji(vae, emojis, config_json["steps"])




