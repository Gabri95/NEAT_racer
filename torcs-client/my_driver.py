from pytocl.driver import Driver
from pytocl.car import State, Command
from model import *
import pickle
from esn_2 import *
import math
import time as tm
import sys
sys.path.insert(0, '../')



class MyDriver(Driver):
    
    def __init__(self, parameters_file=None, name=None, out_file=None):
        super(MyDriver, self).__init__()
        
        with open(parameters_file, 'rb') as f:
            self.net = pickle.load(f)
            self.net.reset()
        
        self.name = name
        self.out_file = out_file

        self.last_lap_time = 0
        self.curr_time = -10.0
        self.time = 0.0
        self.distance = -1.0
        self.distance_from_start = 0.0
        self.laps = -1
        self.damage = 0
        self.offroad_penalty = 0
        self.iterations_count = 0
        self.avg_speed = 0
        self.z = 0
        self.stopped = False
        
        print('Driver initialization completed')
       
    def drive(self, carstate: State) -> Command:
        """
        Produces driving command in response to newly received car state.

        This is a dummy driving routine, very dumb and not really considering a
        lot of inputs. But it will get the car (if not disturbed by other
        drivers) successfully driven along the race track.
        """
        print('Drive')
        input = carstate.to_input_array()
        
        
        if np.isnan(input).any():
            if not self.stopped:
                self.saveResults()
            
            print(input)
            print('NaN INPUTS!!! STOP WORKING')
            return Command()

        self.stopped = np.isnan(input).any()
        
        
        projected_speed = get_projected_speed(carstate.speed_x, carstate.speed_y, carstate.angle)
        print(tm.ctime())
        print('wheel velocities =', carstate.wheel_velocities)
        print('distance raced = ', carstate.distance_raced)
        # print('distances = ', carstate.distances_from_edge)
        print('distance center = ', carstate.distance_from_center)
        
        print('projected speed =', projected_speed)
        print('vx = ', carstate.speed_x)
        print('vy = ', carstate.speed_y)
        print('angle = ', carstate.angle)
        
        # print('self curr_time', self.curr_time)
        # print('self dist from start', self.distance_from_start)
        # print('dist from start', carstate.distance_from_start)
        #
        # print('current lap time: ', carstate.current_lap_time)
        # print('last lap time: ', carstate.last_lap_time)
        # print('self laps: ', self.laps)
        # print('self time', self.time)
        #print('damage = ', carstate.damage)
        #print('rpm = ', carstate.rpm)
        #print('gear = ', carstate.gear)
        print('offroad penalty = ', self.offroad_penalty)
        print('z = ', carstate.z)
        
        
        self.avg_speed += projected_speed
        self.distance = carstate.distance_raced
        #if self.curr_time > carstate.current_lap_time:
        #if self.distance_from_start - carstate.distance_from_start > 100:
        if self.laps >= 0 and self.curr_time > carstate.current_lap_time: #self.last_lap_time != carstate.last_lap_time:
            self.time += carstate.last_lap_time
            self.laps += 1
        elif self.laps < 0 and self.distance_from_start - carstate.distance_from_start > 400:
            self.time += carstate.last_lap_time
            self.laps += 1

        self.last_lap_time = carstate.last_lap_time
        self.distance_from_start = carstate.distance_from_start
        self.curr_time = carstate.current_lap_time
        self.damage = carstate.damage
        
        self.z = carstate.z
        
        self.iterations_count += 1
        self.offroad_penalty += (max(0, math.fabs(carstate.distance_from_center) - 0.98)) ** 2
        
        try:
            output = self.net.activate(input)

            for i in range(len(output)):
                if np.isnan(output[i]):
                    output[i] = 0.0
                else:
                    output[i] = max(output[i], 0.0)

            print('Out = ' + str(output))
            
            command = Command()
            
            # self.accelerate(0.3, 0, 0, carstate, command)
            # self.steer(0, 0, carstate, command)
            
            self.accelerate(output[0], output[1], 0, carstate, command)
            self.steer(output[2], output[3], carstate, command)

        except OverflowError as err:
            print('--------- OVERFLOW! ---------')
            self.saveResults()
            raise err
        
        if self.data_logger:
            self.data_logger.log(carstate, command)

        return command
    
    
    def accelerate(self, acceleration, brake, shift, carstate, command):
    
        command.accelerator = acceleration

        command.gear = carstate.gear
        
        if shift >= 0.5:
            command.gear = carstate.gear or 1
            command.gear = -1*np.sign(command.gear)
            
        if command.gear >= 0 and brake < 0.1 and carstate.rpm > 8000:
            command.gear = min(6, command.gear + 1)

        command.brake = brake

        #if carstate.rpm < 2500:
        if carstate.rpm < 2500 and command.gear > 1:
            command.gear = command.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

    def steer(self, left, right, carstate, command):
        
        command.steering = left - right
        
    def saveResults(self):
        if self.out_file is not None:
            f = open(self.out_file, 'w')
            #f.write("{}: {}, {}".format(self.name, self.distance, self.curr_time))
            f.write("{}, {}, {}, {}, {}, {}, {}".format(
                self.distance,
                self.time + self.curr_time,
                self.laps,
                self.distance_from_start,
                self.damage,
                #self.offroad_penalty / self.iterations_count if self.iterations_count > 0 else 0,
                math.sqrt(self.offroad_penalty / self.iterations_count) if self.iterations_count > 0 else 0,
                self.avg_speed/self.iterations_count if self.iterations_count > 0 else 0)
            )
            f.close()
    
    def on_shutdown(self):
        """
        Server requested driver shutdown.

        Optionally implement this event handler to clean up or write data
        before the application is stopped.
        """
        print('Client ShutDown')
        
        self.saveResults()
        
        if self.data_logger:
            self.data_logger.close()
            self.data_logger = None


def get_projected_speed(speed_x, speed_y, angle):
    
    velocity = get_velocity(speed_x, speed_y)
    return velocity[1]*math.cos(math.radians(angle - velocity[0]))



def get_velocity(speed_x, speed_y):
    return np.angle(speed_x + 1j*speed_y, True), np.sqrt(speed_x**2 + speed_y**2)