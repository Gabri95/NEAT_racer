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
import datetime
from shutil import copyfile
import traceback
import time

sys.path.insert(0, '../')

from neatsociety import nn, population, statistics, visualize


client_path = '../../torcs-client/'
debug_path = '../../debug/'
models_path = '../models/'
results_path = '../../model_results/'
config_path = '../../config/'
shutdown_wait = 1
result_saving_wait = 5
timeout_server = 120


def kill(proc):
    proc.terminate()
    time.sleep(shutdown_wait)
    
    if proc.poll() is None:
        proc.kill()
        time.sleep(shutdown_wait)
        

def evaluate(net):
    dir = os.path.dirname(os.path.realpath(__file__))
    
    current_time = datetime.datetime.now().isoformat()
    start_time = time.time()

    phenotype_file = os.path.join(dir, models_path, "model_{}.pickle".format(current_time))
    results_file = os.path.join(dir, results_path, 'results_{}'.format(current_time))
    
    pickle.dump(net, open(phenotype_file, "wb"))
    
    # subprocess.call(['time',
    #                  'python',
    #                  os.path.join(dir, '../../torcs-server/torcs_tournament.py'),
    #                  '../config/quickrace.yml'])

    client_stdout_path = os.path.join(dir, debug_path, 'client/out.log')
    client_stderr_path = os.path.join(dir, debug_path, 'client/err.log')
    server_stdout_path = os.path.join(dir, debug_path, 'server/out.log')
    server_stderr_path = os.path.join(dir, debug_path, 'server/err.log')
    
    client_stdout = open(client_stdout_path, 'w')
    client_stderr = open(client_stderr_path, 'w')
    server_stdout = open(server_stdout_path, 'w')
    server_stderr = open(server_stderr_path, 'w')
    
    opened_files = [client_stdout, client_stderr, server_stdout, server_stderr]

    timeout = False
    try:
        print('Starting Client')
        client = subprocess.Popen(['./start.sh', '-l', '-p', '3001', '-w', phenotype_file, '-o', results_file],
                                  stdout=client_stdout,
                                  stderr=client_stderr,
                                  cwd=os.path.join(dir, client_path)
                                  )
        
        
        print('Waiting for server to stop')
        subprocess.call(['time',
                         'torcs',
                         '-d',
                         '-r',
                         os.path.join(dir, config_path, 'quickrace.xml')],
                        timeout=timeout_server,
                        stdout=server_stdout,
                        stderr=server_stderr
                        )
    except subprocess.TimeoutExpired:
        print('SERVER TIMED-OUT!')
        timeout = True
        
        copy_path = os.path.join(dir, debug_path, 'model_timedout_{}.pickle'.format(current_time))
        
        print('Copying the model which caused the timeout to:', copy_path)
        copyfile(phenotype_file, copy_path)
        
    except:
        print('Ops! Something happened"')
        traceback.print_exc()

        print('Killing client')
        kill(client)
        
        for file in opened_files:
            file.close()
            
        copyfile(client_stdout_path, os.path.join(dir, debug_path, 'client/ERROR_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(dir, debug_path, 'client/ERROR_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(dir, debug_path, 'server/ERROR_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(dir, debug_path, 'server/ERROR_err_{}.log'.format(current_time)))
        
        raise
    
    
    
    print('Killing client')
    kill(client)

    for file in opened_files:
        file.close()
    
    if timeout:
        copyfile(client_stdout_path, os.path.join(dir, debug_path, 'client/timeout_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(dir, debug_path, 'client/timeout_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(dir, debug_path, 'server/timeout_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(dir, debug_path, 'server/timeout_err_{}.log'.format(current_time)))
    
    
    print('Simulation ended')
    
    # for zippath in glob.iglob(os.path.join(dir, '../*.txt')):
    #     os.remove(zippath)
    # for zippath in glob.iglob(os.path.join(dir, '../../torcs-client/output/*.txt')):
    #     os.remove(zippath)


    #wait a couple of seconds for the results file to be created
    time.sleep(2)
    
    #if the result file hasn't been created yet, try 10 times waiting 'result_saving_wait' seconds between each attempt
    attempts = 0
    while not os.path.exists(results_file) and attempts < 10:
        attempts += 1
        print('Attempt', attempts, 'Time =', datetime.datetime.now().isoformat())
        time.sleep(result_saving_wait)
        
        
    try:
        results = open(results_file, 'r')

        values = [float(x) for x in results.readline().split(',')]

        results.close()
    except IOError:
        print("Can't find the result file!")
        traceback.print_exc()
        values = None

    end_time = time.time()
    
    print('Total Execution Time =', end_time - start_time, 'seconds')
    
    return values


def eval_fitness(genomes):
    
    dir = os.path.dirname(os.path.realpath(__file__))
    
    print('\nStarting evaluation...\n\n')
    
    tot = len(genomes)
    
    for i, g in enumerate(genomes):
        
        print('evaluating', i+1, '/', tot, '\n')
        net = nn.create_recurrent_phenotype(g)
        
        values = evaluate(net)
        
        if values is None:
            fitness = -100
        else:
            distance, duration, laps, distance_from_start, damage, penalty, avg_speed = values[:7]
            
            fitness = distance - 0.1*damage - 100*penalty
            print('\tDistance = ', distance)
            print('\tDamage = ', damage)
            print('\tPenalty = ', penalty)
            print('\tAvgSpeed = ', avg_speed)
            
            if laps >= 3:
                fitness += 200.0*avg_speed#distance/(duration+1)
        
        print('\tFITNESS =', fitness, '\n')
        
        g.fitness = fitness

    print('\n... finished evaluation\n\n')
    print('Cleaning directories')
    
    for zippath in glob.iglob(os.path.join(dir, results_path, 'results_*')):
        os.remove(zippath)
    for zippath in glob.iglob(os.path.join(dir, models_path, '*')):
        os.remove(zippath)
    


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
    
    #build the directories used used by this script if they don't exist
    
    real_dir = os.path.dirname(os.path.realpath(__file__))
    
    directories = [client_path, debug_path + 'client', debug_path + 'server', models_path, results_path, config_path]
    
    for d in directories:
        directory = os.path.join(real_dir, d)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    
    
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
