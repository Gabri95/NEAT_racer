from pytocl.driver import Driver
from pytocl.car import State, Command
from model import *
from my_driver import *
import pickle
import math
import time as tm
import sys

sys.path.insert(0, '../')


class Driver1(MyDriver):
    def __init__(self, parameters_file=None, out_file=None, unstuck=True):
        super(Driver1, self).__init__(parameters_file, out_file, unstuck)
        
    def drive(self, carstate: State) -> Command:
            
        print('Drive')
        input = carstate.to_input_array()
        
        if np.isnan(input).any():
            if not self.stopped:
                self.saveResults()
            
            print(input)
            print('NaN INPUTS!!! STOP WORKING')
            return Command()
        
        self.stopped = np.isnan(input).any()

        self.print_log(carstate)

        self.update(carstate)
        
        try:
            output = self.net.activate(input)
            command = Command()
            
            if self.unstuck and self.isStuck(carstate):
                print('Ops! I got STUCK!!!')
                self.reverse(carstate, command)
            
            else:
                
                if self.unstuck and carstate.gear <= 0:
                    carstate.gear = 1
                
                for i in range(len(output)):
                    if np.isnan(output[i]):
                        output[i] = 0.0
                
                print('Out = ' + str(output))
                
                
                self.accelerate(output[0], output[1], 0, carstate, command)
                
                if len(output) == 3:
                    # use this if the steering is just one output
                    self.steer(output[2], 0, carstate, command)
                else:
                    # use this command if there are 2 steering outputs
                    self.steer(output[2], output[3], carstate, command)
        except:
            print('Error!')
            self.saveResults()
            raise
        
        
        
        if self.data_logger:
            self.data_logger.log(carstate, command)
        
        return command
    