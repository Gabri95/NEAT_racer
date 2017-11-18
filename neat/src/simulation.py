

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


sys.path.insert(0, '../')

from neatsociety import nn, population, statistics, visualize


client_path = '../../torcs-client/'
debug_path = '../../debug/'
models_path = '../models/'
base_results_path = '../../model_results/'
config_path = '../../config/'
shutdown_wait = 10
result_saving_wait = 5
timeout_server = 100

FILE_PATH = os.path.realpath(__file__)
DIR_PATH = os.path.dirname(FILE_PATH)


def evaluate(net,
            configuration,
            port=3001,
            client_path = client_path,
            debug_path = debug_path,
            models_path = models_path,
            results_path = base_results_path,
            config_path = config_path,
            shutdown_wait = shutdown_wait,
            result_saving_wait = result_saving_wait,
            timeout_server = timeout_server):
    
    dir = os.path.dirname(os.path.realpath(__file__))
    
    current_time = datetime.datetime.now().isoformat()
    start_time = time.time()

    phenotype_file = os.path.join(dir, models_path, "model_{}.pickle".format(current_time))
    results_file = os.path.join(dir, results_path, 'results_{}'.format(current_time))
    
    pickle.dump(net, open(phenotype_file, "wb"))
    
    client_stdout_path = os.path.join(dir, debug_path, 'client/out.log')
    client_stderr_path = os.path.join(dir, debug_path, 'client/err.log')
    server_stdout_path = os.path.join(dir, debug_path, 'server/out.log')
    server_stderr_path = os.path.join(dir, debug_path, 'server/err.log')
    
    client_stdout = open(client_stdout_path, 'w')
    client_stderr = open(client_stderr_path, 'w')
    server_stdout = open(server_stdout_path, 'w')
    server_stderr = open(server_stderr_path, 'w')
    
    opened_files = [client_stdout, client_stderr, server_stdout, server_stderr]
    
    server = None

    print('Starting Client')
    client = subprocess.Popen(['./start.sh', '-l', '-p', str(port), '-w', phenotype_file, '-o', results_file],
                              stdout=client_stdout,
                              stderr=client_stderr,
                              cwd=os.path.join(dir, client_path),
                              preexec_fn=os.setsid
                              )

    # wait a few seconds to let client start
    #time.sleep(2)
    
    timeout = False
    try:
        
        print('Waiting for server to stop')
        server = subprocess.Popen(
            ['time',
            'torcs',
            '-nofuel',
            '-r',
            os.path.join(dir, config_path, configuration)],
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
        
        copy_path = os.path.join(dir, debug_path, 'model_timedout_{}.pickle'.format(current_time))
        
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
            
        copyfile(client_stdout_path, os.path.join(dir, debug_path, 'client/ERROR_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(dir, debug_path, 'client/ERROR_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(dir, debug_path, 'server/ERROR_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(dir, debug_path, 'server/ERROR_err_{}.log'.format(current_time)))
        
        raise

    
    print('Killing client')

    #Try to be gentle
    os.killpg(os.getpgid(client.pid), signal.SIGTERM)
    
    #give it some time to stop gracefully
    client.wait(timeout=shutdown_wait)
    
    #if it is still running kill it
    if client.poll() is None:
        print('\tTrying to kill client')
        os.killpg(os.getpgid(client.pid), signal.SIGKILL)
        time.sleep(shutdown_wait)
    
    for file in opened_files:
        file.close()
    
    if timeout:
        copyfile(client_stdout_path, os.path.join(dir, debug_path, 'client/timeout_out_{}.log'.format(current_time)))
        copyfile(client_stderr_path, os.path.join(dir, debug_path, 'client/timeout_err_{}.log'.format(current_time)))
        copyfile(server_stdout_path, os.path.join(dir, debug_path, 'server/timeout_out_{}.log'.format(current_time)))
        copyfile(server_stderr_path, os.path.join(dir, debug_path, 'server/timeout_err_{}.log'.format(current_time)))
    
    
    print('Simulation ended')
    

    #wait a couple of seconds for the results file to be created
    time.sleep(1)
    
    #if the result file hasn't been created yet, try 10 times waiting 'result_saving_wait' seconds between each attempt
    attempts = 0
    while not os.path.exists(results_file) and attempts < 10:
        attempts += 1
        print('Attempt', attempts, 'Time =', datetime.datetime.now().isoformat())
        time.sleep(result_saving_wait)
        
    #try opening the file
    try:
        results = open(results_file, 'r')
        
        #read the comma-separated values in the first line of the file
        values = [float(x) for x in results.readline().split(',')]

        results.close()
        
    except IOError:
        #if the files doesn't exist print, there might have been some error...
        #print the stacktrace and return None
        
        print("Can't find the result file!")
        traceback.print_exc()
        values = None

    end_time = time.time()
    
    print('Total Execution Time =', end_time - start_time, 'seconds')
    
    return values


def initialize_experiments(
            configuration,
            name,
            output_dir,
            port=3001):
    
    
    output_dir = os.path.join(output_dir, name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    results_path = os.path.join(base_results_path, name)
    
    directories = [client_path, os.path.join(debug_path, name, 'client'), os.path.join(debug_path, name,'server'), models_path,
                   results_path, config_path]
    
    for d in directories:
        directory = os.path.join(DIR_PATH, d)
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    if not os.path.isfile(os.path.join(DIR_PATH, config_path, configuration)):
        print('Error! Configuration file "{}" does not exist in {}'.format(configuration, os.path.join(DIR_PATH, config_path)))
        return
    
    return output_dir, results_path, models_path, lambda net: evaluate(net, configuration=configuration, port=port, debug_path=os.path.join(debug_path, name), client_path=client_path, results_path=results_path)