"""
This example produces networks that can remember a fixed-length sequence of bits. It is
intentionally very (overly?) simplistic just to show the usage of the NEAT library. However,
if you come up with a more interesting or impressive example, please submit a pull request!
"""

from __future__ import print_function

import math
import os
import random
import pickle
import numpy as np
import subprocess
import glob
import os.path
import argparse
import sys

sys.path.insert(0, '../')

import neatsociety
from neatsociety import nn, population, statistics, visualize, parallel, activation_functions

def eval_fitness(genomes):
    dir = os.path.dirname(os.path.realpath(__file__))
    
    print('\nStarting evaluation...\n\n')
    
    for g in genomes:
        net = nn.create_recurrent_phenotype(g)
        
        pickle.dump(net, open( "../model.pickle", "wb" ))
    
        subprocess.call(['time',
                         'python',
                         os.path.join(dir,'../../torcs-server/torcs_tournament.py'),
                         '../config/quickrace.yml'])
    
        for zippath in glob.iglob(os.path.join(dir, '../*.txt')):
            os.remove(zippath)
            
        for zippath in glob.iglob(os.path.join(dir, '../../torcs-client/output/*.txt')):
            os.remove(zippath)
        
        result_file = open('../model_results/results', 'r')
    
        values = [float(x) for x in result_file.readline().split(',')]
        
        distance, time, laps, distance_from_start, damage, penalty = values[:6]
        
        fitness = distance - 0.1*damage - 100*penalty
        print('\tDistance = ', distance)
        print('\tDamage = ', damage)
        print('\tPenalty = ', penalty)
        
        if laps >= 3:
            fitness += 200.0*distance/(time+1)
        
        print('\tFITNESS = ' + str(fitness))
        
        g.fitness = fitness

    print('\n... finished evaluation\n\n')


def eval_fitness_test(genomes):
    for i, g in enumerate(genomes):
        g.fitness = i


def get_best_genome(population):
    best = None
    
    for s in population.species:
        for g in s.members:
            if best is None or (g.fitness is not None and g.fitness > best.fitness):
                best = g
    return best


def run(generations=20, frequency=None, output_dir=None, checkpoint=None):
    
    if frequency is None:
        frequency = generations
        
    local_dir = os.path.dirname(__file__)
    pop = population.Population(os.path.join(local_dir, 'nn_config'))
    
    if checkpoint is not None:
        print('Loading from ', checkpoint)
        pop.load_checkpoint(checkpoint)
    
    for g in range(1, generations+1):
        
        pop.run(eval_fitness, 1)
        
        if g % frequency == 0:
            if output_dir is not None:
                print('Saving best net in ../best.pickle')
                pickle.dump(nn.create_recurrent_phenotype(get_best_genome(pop)), open("../best.pickle", "wb"))
                
                new_checkpoint = os.path.join(output_dir, 'neat_gen_{}.checkpoint'.format(pop.generation))
                print('Storing to ', new_checkpoint)
                pop.save_checkpoint(new_checkpoint)
                
                print('Plotting statistics')
                visualize.plot_stats(pop.statistics, filename=os.path.join(output_dir, 'avg_fitness.svg'))
                visualize.plot_species(pop.statistics, filename=os.path.join(output_dir, 'speciation.svg'))
    
    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    print('Saving best net in ../best.pickle')
    pickle.dump(nn.create_recurrent_phenotype(get_best_genome(pop)), open("../best.pickle", "wb"))
    
    # Display the most fit genome.
    print('\nBest genome:')
    winner = pop.statistics.best_genome()
    print(winner)

    

    # Visualize the winner network and plot/log statistics.
    visualize.draw_net(winner, view=True, filename="nn_winner.gv")
    visualize.draw_net(winner, view=True, filename="nn_winner-enabled.gv", show_disabled=False)
    visualize.draw_net(winner, view=True, filename="nn_winner-enabled-pruned.gv", show_disabled=False, prune_unused=True)
    visualize.plot_stats(pop.statistics)
    visualize.plot_species(pop.statistics)
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NEAT algorithm'
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        help='Checkpoint file',
        type=str
    )
    parser.add_argument(
        '-g',
        '--generations',
        help='Number of generations to train',
        type=int,
        default=10
    )

    parser.add_argument(
        '-f',
        '--frequency',
        help='How often to store checkpoints',
        type=int
    )
    
    parser.add_argument(
        '-o',
        '--output_dir',
        help='Directory where to store checkpoint.',
        type=str
    )
    
    args, _ = parser.parse_known_args()
    
    run(**args.__dict__)
