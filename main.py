import os
import json
import random
from datetime import datetime
from time import time

from cmaes import main_cma_es
from utils import create_save_folder

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import argparse
from config import *
from render.pylinhas_renderer import *

render_table = {
    "pylinhas": PylinhasRenderer,
}


def setup_args(resolution, dirs = None):
    parser = argparse.ArgumentParser(description="Evolve to objective")
    IMG_SIZE = resolution[0]

    parser.add_argument('--evolution-type', default=EVOLUTION_TYPE, help='Specify the type of evolution. (cmaes or adam). Default is {}.'.format(EVOLUTION_TYPE))
    parser.add_argument('--random-seed', default=RANDOM_SEED, type=int, help='Use a specific random seed (for repeatability). Default is {}.'.format(RANDOM_SEED))
    parser.add_argument('--save-folder', default=SAVE_FOLDER, help="Directory to experiment outputs. Default is {}.".format(SAVE_FOLDER))
    parser.add_argument('--n-gens', default=N_GENS, type=int, help='Maximum generations. Default is {}.'.format(N_GENS))
    parser.add_argument('--pop-size', default=POP_SIZE, type=int, help='Population size. Default is {}.'.format(POP_SIZE))
    parser.add_argument('--save-all', default=SAVE_ALL, action='store_true', help='Save all Individual images. Default is {}.'.format(SAVE_ALL))
    parser.add_argument('--checkpoint-freq', default=CHECKPOINT_FREQ, type=int, help='Checkpoint save frequency. Default is {}.'.format(CHECKPOINT_FREQ))
    parser.add_argument('--verbose', default=VERBOSE, action='store_true', help='Verbose. Default is {}.'.format(VERBOSE))
    parser.add_argument('--num-lines', default=NUM_LINES, type=int, help="Number of lines. Default is {}".format(NUM_LINES))
    parser.add_argument('--renderer-type', default=RENDERER, help="Choose the renderer. Default is {}".format(RENDERER))
    parser.add_argument('--img-size', default=IMG_SIZE, type=int, help='Image dimensions during testing. Default is {}.'.format(IMG_SIZE))
    parser.add_argument('--target-class', default=TARGET_CLASS, help='Which target classes to optimize. Default is {}.'.format(TARGET_CLASS))
    parser.add_argument("--networks", default=NETWORKS, help="comma separated list of networks (no spaces). Default is {}.".format(NETWORKS))
    parser.add_argument('--target-fit', default=TARGET_FITNESS, type=float, help='target fitness stopping criteria. Default is {}.'.format(TARGET_FITNESS))
    parser.add_argument('--from-checkpoint', default=FROM_CHECKPOINT, help='Checkpoint file from which you want to continue evolving. Default is {}.'.format(FROM_CHECKPOINT))
    parser.add_argument('--sigma', default=SIGMA, type=float, help='The initial standard deviation of the distribution. Default is {}.'.format(SIGMA))
    #parser.add_argument('--clip-model', default=CLIP_MODEL, help='Name of the CLIP model to use. Default is {}. Availables: {}'.format(CLIP_MODEL, clip.available_models()))
    parser.add_argument('--clip-prompts', default=None, help='CLIP prompts to use for the generation. Default is the target class')
    parser.add_argument('--input-image', default=None, help='Image to use as input.')
    parser.add_argument('--adam-steps', default=ADAM_STEPS, type=int, help='Number of steps from Adam. Default is {}.'.format(ADAM_STEPS))
    parser.add_argument('--lr', default=LR, type=float, help='Learning rate for the Adam optimizer. Default is {}.'.format(LR))
    parser.add_argument('--lamarck', default=LAMARCK, action='store_true', help='Lamarck. Default is {}.'.format(LAMARCK))

    args = parser.parse_args()


    if args.from_checkpoint and dirs is not None:
        args.save_folder += dirs[0]
        args.experiment_name = args.from_checkpoint.replace("_checkpoint.pkl", "")
        args.sub_folder = "from_checkpoint"
        save_folder, sub_folder = create_save_folder(args.save_folder, args.sub_folder)
        args.checkpoint = "{}/{}".format(save_folder, args.from_checkpoint)
    else:
        if args.clip_prompts:
            prompt = args.clip_prompts.replace(" ", "_")
        elif args.input_image:
            prompt = args.input_image
        else:
            prompt = args.target_class

        args.experiment_name = f"{args.renderer_type}_L{args.num_lines}_{prompt}_{args.random_seed if args.random_seed else datetime.now().strftime('%Y-%m-%d_%H-%M')}"
        args.sub_folder = f"{args.experiment_name}_{args.n_gens}_{args.pop_size}"

    if args.random_seed:
        print("Setting random seed: ", args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.normalization = (args.renderer_type == "biggan")
    args.renderer = render_table[args.renderer_type](args)


    if args.pop_size <= 1:
        print(f"Population size as {args.pop_size}, changing to Adam.")
        args.evolution_type = "adam"

    return args


def main(fitness_func, se, wd, resolution, population = None, strat = None, debug = False):
    # Get time of start of the program
    if debug: print("starting pylinhas")
    start_time_total = time()

    # Get arguments
    args = setup_args(resolution, dirs = wd)

    # Get time of start of the evolution
    start_time_evo = time()

    # Main program
    tensors, population, strat, fitnesses = main_cma_es(args, fitness_func, se, wd, resolution, last_pop=population, last_strat=strat)

    # Get end time
    end_time = time()

    if debug:
        evo_time = (end_time - start_time_evo)
        total_time = (end_time - start_time_total)
        print("-" * 20)
        print("Evolution elapsed time: {:.3f}".format(evo_time))
        print("Total elapsed time: {:.3f}".format(total_time))
        print("-" * 20)

    return tensors, population, strat, fitnesses

if __name__ == "__main__":
    main(main)

