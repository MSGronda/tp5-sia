import random
import time
import json
from perceptron.functions import *
from perceptron.multi_perceptron import *
from perceptron.optimizers import ADAM
from training_data.font import *



if __name__ == "__main__":


    with open("denoisingconfig.json","r") as f:
        config_json = json.load(f)


   
    
    random.seed(2)
    np.random.seed(2)
    layer_config = [35, 30, 30, 20, 20, 10, 2, 10, 20, 20, 30, 30, 35]
    beta = 1
    optimizer = ADAM
    optimizer_args = [0.001, 0.9, 0.999, 1e-8]
    batch_size = len(fonts)
    autoencoder = MultiPerceptron(layer_config, partial(sigmoid, beta), partial(sigmoid_derivative, beta), optimizer, optimizer_args)
    t1 = time.time()
    min_error = autoencoder.train(50000, fonts, fonts, batch_size)
    t2 = time.time()
    print(min_error, t2-t1)



    if config_json["strategy"] == "bit_flip_with_probability":
        flip_probability_min = config_json["probability_min"]
        flip_probability_max = config_json["probability_max"]
        step = config_json["step"]


        with open("results_denoising.csv","w") as f:
            current_prob = flip_probability_min
            while current_prob <= flip_probability_max:

                noised_fonts = copy.deepcopy(fonts)    

                for vec in noised_fonts:
                    bit_fliping_with_probability(vec,current_prob)


                counter_match = autoencoder.test(noised_fonts, fonts)
                print(f"{current_prob},{counter_match}",file=f)

                current_prob += step
    elif config_json["strategy"] == "bit_fliping_with_n":

        min_n = config_json["n_min"]
        max_n = config_json["n_max"]
        if max_n > 30:
            quit("Max n to high, max 30")


        with open("results_denoising_n.csv","w") as f:
           
            current_n = min_n
            while current_n <= max_n:

                noised_fonts = copy.deepcopy(fonts)    

                for vec in noised_fonts:
                    bit_fliping_with_n(vec,current_n)


                counter_match = autoencoder.test(noised_fonts, fonts)
                print(f"{current_n},{counter_match}",file=f)

                current_n += 1
    else:
        quit("Invalid noising strategy")