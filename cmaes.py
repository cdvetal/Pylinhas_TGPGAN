import os
import pickle
import random
import csv

import torchvision.transforms.functional as TF
import numpy as np
import torch
from deap import base
from deap import cma
from deap import creator
from deap import tools
from torch import optim
from torchvision.utils import save_image
from PIL import Image
from fitnesses import calculate_fitness
from utils import save_gen_best
from config import *

import time

sep = os.path.sep
cur_iteration = 0


def evaluate1(args, individual, dim):
    renderer = args.renderer

    optimizers = renderer.to_adam(individual)

    img = renderer.render(dim[0])
    fitness = calculate_fitness(args.fitnesses, img)
    
    for gen in range(args.adam_steps):
        for optimizer in optimizers:
            optimizer.zero_grad()

        (-fitness).backward()

        for optimizer in optimizers:
            optimizer.step()

        img = renderer.render(dim[0])
        fitness = calculate_fitness(args.fitnesses, img)

        if args.renderer_type == "vdiff" and gen >= 1:
            lr = renderer.sample_state[6][gen] / renderer.sample_state[5][gen]
            renderer.individual = renderer.makenoise(gen)
            renderer.individual.requires_grad_()
            to_optimize = [renderer.individual]
            opt = optim.Adam(to_optimize, lr=min(lr * 0.001, 0.01))
            optimizers = [opt]

        if torch.min(img) < 0.0:
            img = (img + 1) / 2

        save_image(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{cur_iteration}_{gen}.png")

    print(fitness.item())

    if args.lamarck:
        individual[:] = renderer.get_individual()
    return [fitness]


def transform_to_grayscale(rgb, binary = False): # [0, 1]
    gs = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    if binary: gs = (gs > .5).astype(float)
    return gs

def transform_render_to_tensor(render_img): # [0, 1]
    return render_img.squeeze().permute(1, 2, 0).numpy().squeeze()

def transform_tensor_to_domain(img): # [0, 1] to [-1, 1]
    return img * 2.0 - 1.0

def transform_tensor_to_img(img): # [0, 1] to [0, 255]
    return np.clip(img * 255.0, 0.0, 255.0)

def transform_render_to_domain(img, dims = []):
    img = transform_render_to_tensor(img)
    if dims[-1] == 1:
        img = transform_to_grayscale(img)
    return transform_tensor_to_domain(img)

def transform_render_to_img(img, dims = []):
    img = transform_render_to_tensor(img)
    if dims[-1] == 1:
        img = transform_to_grayscale(img)
    return transform_tensor_to_img(img)


def get_pylinahs_tensors(renderer, population, dims, debug=False, filename=""):
    tensors = []
    # get image tensors
    for index, ind in enumerate(population):
        if debug: print("printing index", index)
        _ = renderer.to_adam(ind, gradients=False)

        filename_ind = "{}_ind{}".format(filename, index)
        r_image = renderer.render(dims[0], filename=filename_ind)
        img = transform_render_to_domain(r_image, dims=dims)

        if debug:
            print(np.min(img))
            print(np.max(img))
            print(img.shape)

        tensors.append(img)

    return tensors



def get_pylinahs_tensors_processing(renderer, population, dims, debug = False, filename=""):
    tensors = []
    # get image tensors
    str_genes_pop = "{}\n".format(dims[0])

    #get pop genes
    for index, ind in enumerate(population):
        _ = renderer.to_adam(ind, gradients=False)

        genes_ind = renderer.getGenes()
        #print(genes_ind)
        str_genes_pop += "{}\n".format(genes_ind)

    # write in a txt file the genes
    path = "/Users/jessicaparente/Desktop/Pylinhas_jp/Pylinhas_jp_p/Processing_Generator/data/"
    txt_path = "{}{}.txt".format(path, filename)
    f = open(txt_path, "w")
    f.write(str_genes_pop)
    f.close()

    # get images
    ind_atual = 1
    ind_path = "{}{}_ind{}.png".format(path, filename, ind_atual)


    while (ind_atual - 1) != len(population):
        if os.path.isfile(ind_path):
            print('Evaluating images')
            try:
                processing_image = Image.open(ind_path) #estava do lado do render
                r_image = TF.to_tensor(processing_image).unsqueeze(0) #estava do lado do render

                img = transform_render_to_domain(r_image, dims=dims)
                tensors.append(img)

                os.remove(ind_path)

                # Update Ind
                ind_atual += 1
                ind_path = "{}{}_ind{}.png".format(path, filename, ind_atual)

            except Exception as e:
                print("An exception occurred", e)
        time.sleep(0.5)

    return tensors


@torch.no_grad()
def save_image1(img, fp, dims = None):
    #print("save image beg shape", img.shape)
    if not (2 <= len(dims) <= 3):
        print("Unable to save image: Wrong format!", img.shape)
    aux = np.uint8(img)
    if dims[-1] == 1:
        Image.fromarray(aux, mode="L").save(fp)
    else:
        Image.fromarray(aux, mode="RGB").save(fp)


def main_cma_es(args, fitness_func, se, wd, resolution, last_pop = None, last_strat = None, debug = False):
    if debug: print("resolution: ", resolution)
    global cur_iteration

    renderer = args.renderer

    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    #toolbox.register("evaluate", evaluate, args)

    step = se[1] #step atual
    epoch = se[0] #epoch atual

    if (step%RESTART):
        population = last_pop
        strategy = last_strat
    else:
        last_pop = None
        last_strat = None
        strategy = cma.Strategy(centroid=renderer.generate_individual(), sigma=args.sigma, lambda_=args.pop_size)


    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    halloffame = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    gan_im_d = wd[1]
    best_im_d = wd[2]
    log_d = wd[3]
    images_txts = wd[4]
    check_d = wd[0] + "chekpoints" + sep
    os.makedirs(check_d, exist_ok=True)
    pref = "epoch_" + str(se[0]) + "_step_" + str(se[1])
    args.target_fit = 100000.0


    print("Generation: ", end="")
    for cur_iteration in range(args.n_gens):
        if cur_iteration % 10 == 0: print(str(cur_iteration) + " ", end="")

        path = f"epoch{epoch}_step{step}_cur{cur_iteration}" #path txt e image

        # Generate a new population ver mais tarde
        if (cur_iteration == 0) and (last_pop is not None):
            population = last_pop
        else:
            if (cur_iteration == 0):
                population = toolbox.generate()
            else:
                population = toolbox.generate()

        # Evaluate the individuals (fitness wrap)
        tensors = get_pylinahs_tensors(renderer, population, resolution, debug = False, filename=path)
        fitnesses = fitness_func(tensors=tensors)

        # write fitnesses
        fn = wd[0] + "fitnesses.txt"
        with open(fn, mode='a', newline='') as file:
            fwriter = csv.writer(file, delimiter=',')
            fwriter.writerow(fitnesses)

        for ind, fit in zip(population, fitnesses):
            #fit = fit[0].cpu().detach().numpy()
            ind.fitness.values = [fit]

        if args.save_all:
            for index, ind in enumerate(population):
                _ = renderer.to_adam(ind, gradients=False)
                img = transform_render_to_img(tensors[ind], dims = resolution)#renderer.render(resolution), dims = resolution)

                save_image1(img, f"{args.save_folder}/{args.sub_folder}/{args.experiment_name}_{cur_iteration}_{index}.png", dims = resolution)


        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        # Update the hall of fame and the statistics with the
        # currently evaluated population
        halloffame.update(population)
        record = stats.compile(population)
        logbook.record(evals=len(population), gen=cur_iteration, **record)

        if args.verbose:
            print(logbook.stream)


        if halloffame is not None:
            save_gen_best(log_d + pref, "", "", [cur_iteration, halloffame[0], halloffame[0].fitness.values, "_"])
            if debug: print("Best individual:", halloffame[0].fitness.values)
            _ = renderer.to_adam(halloffame[0], gradients=False)

            img = transform_render_to_img(renderer.render(image_size=512, filename=path), dims = resolution)
            save_image1(img, f"{best_im_d}cur_{cur_iteration}_best.png", dims = resolution)



        if halloffame[0].fitness.values[0] >= args.target_fit:
            print("Reached target fitness.\nExiting")
            break

        if cur_iteration % args.checkpoint_freq == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=cur_iteration, halloffame=halloffame, logbook=logbook,
                      np_rndstate=np.random.get_state(), rndstate=random.getstate())
            with open(check_d + "gen_" + str(cur_iteration), "wb") as cp_file:
                pickle.dump(cp, cp_file)
    print()

    tensors = get_pylinahs_tensors(renderer, population, dims = resolution, debug = False, filename=path)

    del creator.FitnessMax
    del creator.Individual

    return tensors, population, strategy, fitnesses