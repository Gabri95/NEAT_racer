
from __future__ import print_function

import os
import pickle
import glob
import os.path
import argparse
import sys
import simulation
import datetime

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


sys.path.insert(0, os.path.join(DIR_PATH, '../../'))

from neatsociety import nn, population, statistics, visualize


def eval_fitness(genomes, evaluate_function=None, cleaner=None):
    
    print('\nStarting evaluation...\n\n')
    
    tot = len(genomes)
    
    #evaluate the genotypes one by one
    for i, g in enumerate(genomes):
        
        print('evaluating', i+1, '/', tot, '\n')
        net = nn.create_recurrent_phenotype(g)
        
        
        #run the simulation to evaluate the model
        values = evaluate_function(net)
        
        if values is None:
            fitness = -100
        else:
            distance, duration, laps, distance_from_start, damage, penalty, avg_speed = values[:7]
            
            fitness = distance - 0.03*damage - 100*penalty
            print('\tDistance = ', distance)
            print('\tDamage = ', damage)
            print('\tPenalty = ', penalty)
            print('\tAvgSpeed = ', avg_speed)
            
            if laps >= 2:
                fitness += 50.0*avg_speed#distance/(duration+1)
        
        print('\tFITNESS =', fitness, '\n')
        
        g.fitness = fitness

    print('\n... finished evaluation\n\n')
    
    if cleaner is not None:
        #at the end of the generation, clean the files we don't need anymore
        cleaner()



def clean_temp_files(results_path, models_path):
    print('Cleaning directories')
    for zippath in glob.iglob(os.path.join(DIR_PATH, results_path, 'results_*')):
        os.remove(zippath)
    for zippath in glob.iglob(os.path.join(DIR_PATH, models_path, '*')):
        os.remove(zippath)
    



def get_best_genome(population):
    best = None
    
    for s in population.species:
        for g in s.members:
            if best is None or best.fitness is None or (g.fitness is not None and g.fitness > best.fitness):
                best = g
    return best


def run(neat_config, name, generations=20, port=3001, frequency=None, output_dir=None, checkpoint=None, configuration=None):

    
    
    if configuration is None:
        print('Error! No configuaration file has been set')
        return

    output_dir, results_path, models_path, EVAL_FUNCTION = simulation.initialize_experiments(configuration, name, output_dir, port=port)
    
    EVAL_POPULATION = lambda pop: eval_fitness(pop, evaluate_function=EVAL_FUNCTION, cleaner=lambda: clean_temp_files(results_path, models_path))
    
    best_model_file = os.path.join(output_dir, 'best.pickle')
    
    if frequency is None:
        frequency = generations
    
    pop = population.Population(neat_config)
    
    if checkpoint is not None:
        print('Loading from ', checkpoint)
        pop.load_checkpoint(checkpoint)
    
    for g in range(1, generations+1):
        
        pop.run(EVAL_POPULATION, 1)
        
        if g % frequency == 0:
            if output_dir is not None:
                print('Saving best net in {}'.format(best_model_file))
                best_genome = get_best_genome(pop)
                pickle.dump(nn.create_recurrent_phenotype(best_genome), open(best_model_file, "wb"))
                
                new_checkpoint = os.path.join(output_dir, 'neat_gen_{}.checkpoint'.format(pop.generation))
                print('Storing to ', new_checkpoint)
                pop.save_checkpoint(new_checkpoint)
                
                print('Plotting statistics')
                visualize.plot_stats(pop.statistics, filename=os.path.join(output_dir, 'avg_fitness.svg'))
                visualize.plot_species(pop.statistics, filename=os.path.join(output_dir, 'speciation.svg'))
                
                print('Save network view')
                visualize.draw_net(best_genome, view=False,
                                   filename=os.path.join(output_dir, "nn_winner-enabled-pruned.gv"),
                                   show_disabled=False, prune_unused=True)

                visualize.draw_net(best_genome, view=False, filename=os.path.join(output_dir, "nn_winner.gv"))
                visualize.draw_net(best_genome, view=False, filename=os.path.join(output_dir, "nn_winner-enabled.gv"),
                                   show_disabled=False)
                
                
    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    print('Saving best net in {}'.format(best_model_file))
    pickle.dump(nn.create_recurrent_phenotype(get_best_genome(pop)), open(best_model_file, "wb"))
    
    # Display the most fit genome.
    #print('\nBest genome:')
    winner = pop.statistics.best_genome()
    #print(winner)

    

    # Visualize the winner network and plot/log statistics.
    visualize.draw_net(winner, view=True, filename=os.path.join(output_dir, "nn_winner.gv"))
    visualize.draw_net(winner, view=True, filename=os.path.join(output_dir, "nn_winner-enabled.gv"), show_disabled=False)
    visualize.draw_net(winner, view=True, filename=os.path.join(output_dir, "nn_winner-enabled-pruned.gv"), show_disabled=False, prune_unused=True)
    visualize.plot_stats(pop.statistics, filename=os.path.join(output_dir, 'avg_fitness.svg'))
    visualize.plot_species(pop.statistics, filename=os.path.join(output_dir, 'speciation.svg'))
    statistics.save_stats(pop.statistics, filename=os.path.join(output_dir, 'fitness_history.csv'))
    statistics.save_species_count(pop.statistics, filename=os.path.join(output_dir, 'speciation.csv'))
    statistics.save_species_fitness(pop.statistics, filename=os.path.join(output_dir, 'species_fitness.csv'))


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

    parser.add_argument(
        '-x',
        '--configuration',
        help='XML configuration file for running the race. It has to be in the {} directory'.format(simulation.config_path),
        type=str,
        default='quickrace.xml'
    )

    parser.add_argument(
        '-p',
        '--port',
        help='Port to use for comunication between server (simulator) and client',
        type=int,
        default=3001
    )

    parser.add_argument(
        '-e',
        '--name',
        help='Experiment name',
        type=str,
        default=datetime.datetime.now().isoformat()
    )

    local_dir = os.path.dirname(__file__)
    default_neat_config = os.path.join(local_dir, 'nn_config')

    parser.add_argument(
        '-n',
        '--neat_config',
        help='NEAT configuration file',
        type=str,
        default=default_neat_config,
    )
    
    args, _ = parser.parse_known_args()
    
    run(**args.__dict__)
