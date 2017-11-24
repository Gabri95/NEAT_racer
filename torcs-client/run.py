#! /usr/bin/env python3

from pytocl.main import main
from driver1 import Driver1
from driver2 import Driver2
from my_driver import MyDriver
from pytocl.driver import Driver
import argparse
import sys
import traceback
import signal


driver = None


def sigterm_handler(_signo, _stack_frame):
    print('Someone killed me')
    global driver
    if driver is not None and isinstance(driver, MyDriver):
        driver.saveResults()
    sys.exit(0)


signal.signal(signal.SIGINT, sigterm_handler)
signal.signal(signal.SIGTERM, sigterm_handler)

registry = {'Driver1': Driver1,
           'Driver2': Driver2}


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
    
    parser.add_argument(
        '-o',
        '--output_file',
        help='File where to print results.',
        type=str
    )

    parser.add_argument(
        '-d',
        '--driver',
        help='Set the type of driver to use',
        type=str,
        default='Driver1'
    )

    parser.add_argument(
        '-u',
        '--unstuck',
        help='Make the drivers automatically try to unstuck',
        action='store_true'
    )
    
    args, _ = parser.parse_known_args()
    
    print(args.parameters_file)
    print(args.output_file)
    print(args.driver)
    
    
    if args.parameters_file is not None:
        driver = registry[args.driver](args.parameters_file, out_file=args.output_file, unstuck=args.unstuck)
    else:
        driver = Driver()
    
    try:
        main(driver)
        
    except Exception as exc:
        traceback.print_exc()

        if args.parameters_file is not None:
            driver.saveResults()
            
        raise
    
    
