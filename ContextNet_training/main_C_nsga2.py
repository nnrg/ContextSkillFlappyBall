
from FlappyBird_4D import Game
from neural_net_torch import Context_Skill_Net as CSnet # Context-Skill Model
from neural_net_torch import Skill_only_Net as S_o_net # Skill-only Model
from neural_net_torch import Context_only_Net as C_o_net # Context-only Model

import os
import copy
import array
import random
import pickle
import numpy as np
from scoop import futures
from subprocess import call
from tensorboardX import SummaryWriter
import torch

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

#
os.environ['SDL_AUDIODRIVER'] = 'dsp'

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

#-----------------------------------------------------------------------------
# Hyperparameters

NGEN = 250
MU = 96
CXPB = 0.9
REPEATS = 5
BOUND_LOW, BOUND_UP = -5.0, 5.0

n_obs = 6
n_actions = 2
net_sample = C_o_net(n_obs, n_actions) #C2 (with memory reset every task)
NDIM = net_sample.computeTotalNumberOfParameters()

base_flap, base_gravity, base_fwd, base_drag = -12., 1., 5., 1. 
percent = 0.2

game_params = {"MAX_TIME"     : 500, # Game will end after MAX_TIME
               "PIPE_SPACING" : 160, # Fixed for now, but it can be variable
               "PIPE_GAP"     : 130, # 130Fixed for now, but it can be variable
               "PIPE_WIDTH"   : 60, # 60Fixed for now, but it can be variable
               "PIPE_GAP_POS" : 216, # Start of the gap
               "FLAP"         : base_flap,
               "GRAVITY"      : base_gravity,
               "FWD"          : base_fwd,
               "DRAG"         : base_drag,
               "BIRD_RADIUS"  : 14,
               "Active_Tasks" : [4],    
               "Param_Range"  : [[0,    0],     # 0) dummy variable
                                 [120,  200],   # 1) PIPE_SPACING 
                                 [20,   150],   # 2) PIPE_WIDTH
                                 [90,   170],   # 3) PIPE_GAP
                                 [100,  332],   # 4) PIPE GAP POSITION
                                 [base_flap * (1-percent), base_flap * (1+percent)],  # 5) BIRD_flap
                                 [base_gravity * (1-percent), base_gravity * (1+percent)],  # 6) GRAVITY
                                 [base_fwd * (1-percent), base_fwd * (1+percent)],  # 7) BIRD_Forward_Flap
                                 [base_drag * (1-percent), base_drag * (1+percent)],  # 8) DRAG
                                 [10.0, 18.0]]} # 9) BIRD_RADIUS 

#-----------------------------------------------------------------------------
# Helper functions

def genotype_to_phenotype(vector):
    # net_sample is needed to build the net from the vector
    net_copy = copy.deepcopy(net_sample)
    vector_copy = copy.deepcopy(np.array(vector))
    
    for p in net_copy.parameters():
        len_slice = p.numel()
        replace_with = vector_copy[0:len_slice].reshape(p.data.size())
        p.data = torch.from_numpy( replace_with )
        vector_copy = np.delete(vector_copy, np.arange(len_slice))    

    return net_copy


def phenotype_to_genotype(net):
    vector = []
    for p in net.parameters():
        vector.append( p.data.numpy().flatten() )
    vector = np.concatenate(vector)
    return vector.tolist() 


def prepare_params():
    # Output: params = [seed, flap, gravity, fwd, drag] * REPEATS
    params = []
    
    flap = game_params["FLAP"]
    gravity = game_params["GRAVITY"]
    fwd = game_params["FWD"]
    drag = game_params["DRAG"]

    flap_list = np.random.uniform(game_params["Param_Range"][5][0], 
                                  game_params["Param_Range"][5][1], REPEATS)
    gravity_list = np.random.uniform(game_params["Param_Range"][6][0], 
                                  game_params["Param_Range"][6][1], REPEATS)
    fwd_list = np.random.uniform(game_params["Param_Range"][7][0], 
                                  game_params["Param_Range"][7][1], REPEATS)
    drag_list = np.random.uniform(game_params["Param_Range"][8][0], 
                                  game_params["Param_Range"][8][1], REPEATS)
    seed_list = np.random.randint(2**32, size=REPEATS, dtype='int64')
    
    for task in range(4): # 4 tasks
        if task == 0:
            for rep in range(REPEATS):
                params.append([seed_list[rep], flap_list[rep], gravity, fwd, drag])
        elif task == 1:
            for rep in range(REPEATS):
                params.append([seed_list[rep], flap, gravity_list[rep], fwd, drag])
        elif task == 2:
            for rep in range(REPEATS):
                params.append([seed_list[rep], flap, gravity, fwd_list[rep], drag])
        elif task == 3:
            for rep in range(REPEATS):
                params.append([seed_list[rep], flap, gravity, fwd, drag_list[rep]])
    return params


def eval_Tasks(ind):
    fits0, fits1 = [], []
    net = genotype_to_phenotype(ind)
    params = ind.params
    REPEATS = int(len(params)/4)
    count = 0
    # [Task-1: flap, Task-2: Gravity, Task-3: Fwd-Flap, Task-4: Drag]
    for task in range(4):
        # net.h_prev = torch.zeros(1, net.context_LSTMcell_size, dtype=torch.double)
        # net.c_prev = torch.zeros(1, net.context_LSTMcell_size, dtype=torch.double)
        for rep in range(REPEATS):
            f0, f1 = Game(net, game_params, params[count])
            fits0.append(f0)
            fits1.append(f1)
            count += 1
    f0_mean = np.mean(fits0) # Average distance
    f1_mean = np.mean(fits1) # Average number of hits
    return f0_mean, f1_mean

#-----------------------------------------------------------------------------

creator.create("FitnessMaxMin", base.Fitness, weights=(1.0, -1.0)) 
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMaxMin, params=None)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.randn)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, NDIM)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_Tasks)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1.0/NDIM)
toolbox.register("select", tools.selNSGA2)
toolbox.register("map", futures.map)

def main(pickle_file=None):
    gen = 0
    stop_run = False
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", np.mean, axis=0)
    # stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "min", "max"
    
    pop = toolbox.population(n=MU)

    if pickle_file:
        cp = pickle.load(open(pickle_file,"rb"))
        pop = cp["population"]
        logbook = cp["logbook"]
        gen = cp["generation"]
        random.setstate(cp["rndstate"])

    # Evaluate the individuals with an invalid fitness
    params = prepare_params()
    invalid_ind = [ind for ind in pop if not ind.fitness.valid] # Genotype
    for ind in invalid_ind:
        ind.params = params
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    # for ind in invalid_ind:
    #     fitness = toolbox.evaluate(ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    
    record = stats.compile(pop)
    logbook.record(gen=gen, evals=len(invalid_ind), **record)
    print(logbook.stream)

    cp = dict(population=pop, generation=gen, logbook=logbook, rndstate=random.getstate())
    with open("gen%d_C_checkpoint.pkl" % (gen), "wb") as cp_file:
        pickle.dump(cp, cp_file)

    # Begin the generational process
    for gen in range(gen+1, NGEN+1):
        if not stop_run:
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2)
                
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)
                del ind1.fitness.values, ind2.fitness.values
        
            # Evaluate the individuals with an invalid fitness (with the same seed no.)
            params = prepare_params()
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.params = params
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            for ind in pop:
                fit = ind.fitness.values
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)

            cp = dict(population=pop, generation=gen, logbook=logbook, rndstate=random.getstate())
            with open("gen%d_C_checkpoint.pkl" % (gen), "wb") as cp_file:
                pickle.dump(cp, cp_file)
                
            if gen > 0:
                call("rm gen%d_*" % (gen-1), shell=True)

            if (gen == NGEN) or (stop_run == True):
                pop_as_nets = []
                for ind in pop:
                    net = genotype_to_phenotype(ind)
                    net.fitness = ind.fitness.values
                    pop_as_nets.append( net )
                break

    # print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))
    print("Final population hypervolume is %f" % hypervolume(pop))
    
    return pop_as_nets

if __name__ == "__main__":

    pop = main()
    pickle.dump(pop, open("popAsNets_C.p","wb"))

