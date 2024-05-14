import datetime
import os
from time import time
from cmaes import main_cma_es
import main
import numpy as np

# NIMA classifier imports
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as preprocess_input_mob
from nima_utils.score_utils import mean_score, std_score

delimiter = os.path.sep


def nima_classifier(**kwargs):
    tensors = kwargs.get('tensors')
    fitnesses = list()

    number_tensors = len(tensors)
    # NIMA classifier
    x = np.stack([tensors[index] for index in range(number_tensors)], axis = 0)
    x = preprocess_input_mob(x)
    scores = model.predict(x, batch_size = number_tensors, verbose=0)

    for index in range(number_tensors):
        mean = mean_score(scores[index])
        std = std_score(scores[index])
        # fit = mean - std
        fit = mean

        fitnesses.append(fit)

    return fitnesses



def create_dirs(args, prefix = None, sub_wd = "runs", run_folder = None):
    # dir names
    date = datetime.datetime.utcnow().strftime('%Y_%m_%d__%H_%M_%S_%f')[:-3]
    filename = "__run__" + date + "__" + str(int(time() * 1000.0) << 16) + "_" + (str(args.random_seed) + "_" if (args.random_seed is not None) else "")

    # paths
    run_folder = run_folder if run_folder is not None else prefix + "_" + filename
    run_dir = os.getcwd() + delimiter + sub_wd + delimiter + prefix + delimiter + run_folder + delimiter
    gan_images = run_dir + "dcgan_images" + delimiter
    best_im_dir = gan_images + delimiter + "bests" + delimiter
    log_dir = run_dir + "logs" + delimiter
    images_to_evaluate = gan_images + "images_to_evaluate" + delimiter

    try:
        os.makedirs(gan_images, exist_ok=True)
        os.makedirs(best_im_dir, exist_ok=True)
        os.makedirs(images_to_evaluate, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    except OSError as error:
        if error is PermissionError:
            print("[WARNING]:\tPermission denied while creating experiment subdirectories.")
        else:
            print("[WARNING]:\tOSError while creating experiment subdirectories.")

    dirs = [run_dir, gan_images, best_im_dir, log_dir, images_to_evaluate]
    return dirs



if __name__ == "__main__":

    debug = True

    # Initialize NIMA classifier
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    model = Model(base_model.input, x)
    model.load_weights('weights/weights_mobilenet_aesthetic_0.07.hdf5')

    # Get arguments
    resolution = [100, 100, 3]
    args = main.setup_args(resolution)
    pref = "nima_example"

    # Create quick
    wd = create_dirs(args, prefix=pref)
    fitness_func = nima_classifier
    args.n_gens = 50 # generations

    # Get time of start of the evolution
    start_time_evo = time()

    # Main program
    if debug: print("Starting evolution")
    tensors, population, strat, fitnesses = main_cma_es(args, fitness_func, [float('inf'), 0], wd, resolution)

    # Get end time
    end_time = time()

    if debug:
        evo_time = (end_time - start_time_evo)
        print("Evolution over!")
        print("-" * 30)
        print("Generations run:", args.n_gens)
        npf = np.array(fitnesses)
        print("Best last pop fitness: ", np.max(npf))
        print("Avg last pop fitness: ", np.mean(npf))
        print("Std last pop fitness: ", np.std(npf))
        print("Evolution time: {:.3f}".format(evo_time))
        print("-" * 30)

