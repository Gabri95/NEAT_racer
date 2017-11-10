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
from neatsociety.reporting import StdOutReporter

best = 0


def eval_fitness(genomes):
    
    global best

    dir = os.path.dirname(os.path.realpath(__file__))
    
    print('\nStarting evaluation...\n\n')
    
    for g in genomes:
        net = nn.create_recurrent_phenotype(g)
        
        pickle.dump(net, open( "../model.pickle", "wb" ))
    
        subprocess.call(['time', 'python', os.path.join(dir,'../../torcs-server/torcs_tournament.py'), '../config/quickrace.yml'])
    
        for zippath in glob.iglob(os.path.join(dir, '../*.txt')):
            os.remove(zippath)
            
        for zippath in glob.iglob(os.path.join(dir, '../../torcs-client/output/*.txt')):
            os.remove(zippath)
        
        result_file = open('../model_results/results', 'r')
    
        distance, time, laps, distance_from_start, damage = [float(x) for x in result_file.readline().split(',')]
        
        fitness = distance - 0.1*damage
        print('Damage = ', damage)
        if laps >= 3:
            fitness += 100.0*(laps*5784.10 + distance_from_start)/(time+1)
        
        if fitness > best:
            pickle.dump(net, open( "../best.pickle", "wb" ))
            best = fitness
        
        print('FITNESS = ' + str(fitness))
        
        g.fitness = fitness

    print('\n... finished evaluation\n\n')


def run():
    
    local_dir = os.path.dirname(__file__)
    pop = population.Population(os.path.join(local_dir, 'nn_config'))

    #pop.add_reporter(StdOutReporter())
    
    #pe = parallel.ParallelEvaluator(eval_fitness, num_workers=1)
    #pop.run(pe.evaluate, 30)
    pop.run(eval_fitness, 2)
    
    print('Number of evaluations: {0}'.format(pop.total_evaluations))

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
    run()
