#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from pytocl.driver import Driver
from model import Model
import argparse
import sys
from shutil import copyfile
import time as tm
import traceback
import signal
import time


driver = None

def sigterm_handler(_signo, _stack_frame):
    print('Someone killed me')
    global driver
    if driver is not None and isinstance(driver, MyDriver):
        driver.saveResults()
    sys.exit(0)


signal.signal(signal.SIGINT, sigterm_handler)
signal.signal(signal.SIGTERM, sigterm_handler)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Client for TORCS racing car simulation with SCRC network'
                    ' server.'
    )
    parser.add_argument(
        '-w',
        '--parameters_file',
        help='Model parameters.',
        type=str
    )
    # parser.add_argument(
    #     '-n',
    #     '--name',
    #     help='Model name.',
    #     type=str
    # )
    parser.add_argument(
        '-o',
        '--output_file',
        help='File where to print results.',
        type=str
    )
    
    parser.add_argument(
        '-l',
        '--print',
        help='Print logs instead of saving to file',
        action='store_true'
    )
    
    args, _ = parser.parse_known_args()
    
    print(args.parameters_file)
    print(args.output_file)
    # print(args.name)

    
    if not args.print:
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        file_out = open('../debug/client/out.log', 'w')
        file_err = open('../debug/client/err.log', 'w')
        sys.stdout = file_out
        sys.stderr = file_err



    if args.parameters_file is not None:
        driver = MyDriver(args.parameters_file, out_file=args.output_file)
    else:
        driver = Driver()
    
    try:
        
        main(driver)
    except Exception as exc:
        traceback.print_exc()

        if args.parameters_file is not None:
            driver.saveResults()
        
        if not args.print:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            file_out.close()
            file_err.close()

            copyfile('../debug/client/out.log', '../debug/client/out_{}.log'.format(tm.time()))
            copyfile('../debug/client/err.log', '../debug/client/err_{}.log'.format(tm.time()))
        
        raise
    
    if not args.print:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        file_out.close()
        file_err.close()
    

    
