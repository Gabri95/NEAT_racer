

from __future__ import print_function

import os
import pickle
import subprocess
import os.path
import sys
import datetime
from shutil import copyfile
import traceback
import time
import signal
import glob

sys.path.insert(0, '../')

#sys.path.insert(0, os.path.dirname(fitness_implementation))

#exec('from ' + os.path.filename(fitness_implementation) + ' import fitness')


FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)

client_path = os.path.join(DIR_PATH, '../../torcs-client/')
shutdown_wait = 10
result_saving_wait = 1
timeout_server = 100


def evaluate(net,
             configuration,
             debug_path,
             models_path,
             results_path,
             port=3001,
             client_path = client_path,
             shutdown_wait = shutdown_wait,
             result_saving_wait = result_saving_wait,
             timeout_server = timeout_server,
             unstuck=False):
    
    
    
    current_time = datetime.datetime.now().isoformat()
    start_time = time.time()

    
    
    results_file = os.path.join(results_path, 'results_{}'.format(current_time))
    phenotype_file = os.path.join(models_path, "model_{}.pickle".format(current_time))
    
    pickle.dump(net, open(phenotype_file, "wb"))
    
    print('Results at', results_file)
    
    client_stdout_path = os.path.join(debug_path, 'client/out.log')
    client_stderr_path = os.path.join(debug_path, 'client/err.log')
    server_stdout_path = os.path.join(debug_path, 'server/out.log')
    server_stderr_path = os.path.join(debug_path, 'server/err.log')
    
    
    client_stdout = open(client_stdout_path, 'w')
    client_stderr = open(client_stderr_path, 'w')
    server_stdout = open(server_stdout_path, 'w')
    server_stderr = open(server_stderr_path, 'w')
    
    opened_files = [client_stdout, client_stderr, server_stdout, server_stderr]
    
    server = None

    print('Starting Client')
    client = subprocess.Popen(['./start.sh', '-p', str(port), '-w', phenotype_file, '-o', results_file, '-d', 'Driver2']
                                    + (['-u'] if unstuck else []),
                              stdout=client_stdout,
                              stderr=client_stderr,
                              cwd=client_path,
                              preexec_fn=os.setsid
                              )
    
    
    # wait a second to let client start
    #time.sleep(1)
    
    timeout = False
    try:
        
        print('Waiting for server to stop')
        server = subprocess.Popen(
                                ['time',
                                'torcs',
                                '-nofuel',
                                '-nolaptime',
                                '-r',
                                configuration],
                                stdout=server_stdout,
                                stderr=server_stderr,
                                preexec_fn=os.setsid
                                )
        
        server.wait(timeout=timeout_server)
    
    except subprocess.TimeoutExpired:
        print('SERVER TIMED-OUT!')
        timeout = True
        
        if server is not None:
            print('Killing server and its children')
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
        
        copy_path = os.path.join(debug_path, 'model_timedout_{}.pickle'.format(current_time))
        
        print('Copying the model which caused the timeout to:', copy_path)
        copyfile(phenotype_file, copy_path)
    except:
        print('Ops! Something happened"')
        traceback.print_exc()

        if server is not None:
            print('Killing server and its children')
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)

        client.terminate()
        time.sleep(1)
        if client.poll() is None:
            client.kill()
        
        for file in opened_files:
            file.close()
            
        copyfile(client_stdout_path, os.path.join(debug_path, 'client/ERROR_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(debug_path, 'client/ERROR_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(debug_path, 'server/ERROR_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(debug_path, 'server/ERROR_err_{}.log'.format(current_time)))
        
        raise

    
    print('Killing client')

    #Try to be gentle
    os.killpg(os.getpgid(client.pid), signal.SIGTERM)
    
    #give it some time to stop gracefully
    client.wait(timeout=shutdown_wait)
    
    #if it is still running, kill it
    if client.poll() is None:
        print('\tTrying to kill client')
        os.killpg(os.getpgid(client.pid), signal.SIGKILL)
        time.sleep(shutdown_wait)
    
    for file in opened_files:
        file.close()
    
    if timeout:
        copyfile(client_stdout_path, os.path.join(debug_path, 'client/timeout_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(debug_path, 'client/timeout_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(debug_path, 'server/timeout_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(debug_path, 'server/timeout_err_{}.log'.format(current_time)))
    
    
    print('Simulation ended')
    

    #wait a second for the results file to be created
    time.sleep(2)
    
    #if the result file hasn't been created yet, try 10 times waiting 'result_saving_wait' seconds between each attempt
    attempts = 0
    while not os.path.exists(results_file) and attempts < 10:
        attempts += 1
        print('Attempt', attempts, 'Time =', datetime.datetime.now().isoformat())
        time.sleep(result_saving_wait + attempts -1)
        
    #try opening the file
    try:
        results = open(results_file, 'r')

        values = []
        
        for line in results.readlines():
            #read the comma-separated values in the first line of the file
             values.append([float(x) for x in line.split(',')])
        
        results.close()
    
    except IOError:
        #if the files doesn't exist print, there might have been some error...
        #print the stacktrace and return None
        
        print("Can't find the result file!")
        traceback.print_exc()

        copyfile(client_stdout_path, os.path.join(debug_path, 'client/ERROR_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(debug_path, 'client/ERROR_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(debug_path, 'server/ERROR_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(debug_path, 'server/ERROR_err_{}.log'.format(current_time)))
        
        values = None

    end_time = time.time()
    
    print('Total Execution Time =', end_time - start_time, 'seconds')
    
    return values




def clean_temp_files(results_path, models_path):
    print('Cleaning directories')
    for zippath in glob.iglob(os.path.join(DIR_PATH, results_path, 'results_*')):
        os.remove(zippath)
    for zippath in glob.iglob(os.path.join(DIR_PATH, models_path, '*')):
        os.remove(zippath)



def initialize_experiments(
            output_dir,
            configuration=None,
            port=3001,
            unstuck=False):
        
    results_path = os.path.join(output_dir, 'results')
    models_path = os.path.join(output_dir, 'models')
    debug_path = os.path.join(output_dir, 'debug')
    checkpoints_path = os.path.join(output_dir, 'checkpoints')
    
    directories = [checkpoints_path,
                   os.path.join(debug_path, 'client'),
                   os.path.join(debug_path, 'server'),
                   models_path,
                   results_path]
    
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
    
    if configuration is None:
        configuration = os.path.join(output_dir, 'configuration.xml')
        
    configuration = os.path.realpath(configuration)
    debug_path = os.path.realpath(debug_path)
    results_path = os.path.realpath(results_path)
    models_path = os.path.realpath(models_path)
    
    
    if not os.path.isfile(configuration):
        print('Error! Configuration file "{}" does not exist in {}'.format(configuration))
        raise FileNotFoundError('Error! Configuration file "{}" does not exist in {}'.format(configuration))
    
    
    eval = lambda net, unstuck=unstuck: evaluate(net, configuration=configuration, unstuck=unstuck, port=port,
                                                         debug_path=debug_path,
                                                         results_path=results_path, models_path=models_path)
        
    return results_path, models_path, debug_path, checkpoints_path, eval