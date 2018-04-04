#!/usr/bin/env python3

import argparse
import numpy as np
import logging
import time
import yaml

from robot import Robot
from robot_motion import RobotMotion
from vis import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "robot_config", 
    type=str,
    help="Configuration file for the robot."
)
parser.add_argument(
    "-r", "--random",
    action="store_true",
    default=False,
    help="Use random number generation rather than a fixed seed."
)
args = parser.parse_args()


def initialize_robots_and_motions(config):
    """Initialize robots and robot motions.

    For each robot, a RobotMotion object will be created. The RobotMotion object
    will store the ground truth sampled from the mean and covariance specified
    in the robot configuration. The Robot will receive the actual mean and
    covariance specified in the initialization file. The number of robots will
    be determined by the configuration file, and they'll be initialized round
    robin from the parameters.

    Args:
        config: Configuration file.

    Returns:
        (robots, motions) tuple of type (list of Robots, list of RobotMotions)
    """

    num_robots = config['num_robots']
    num_parameters = len(config['robot_parameters'])
    logging.debug("Initializing %d Robots, RobotMotions from %d parameters", 
            num_robots, num_parameters) 

    robots = []
    motions = []
    for i in range(num_robots):
        robot_config = config['robot_parameters'][i % num_parameters].copy()
        robot_config['id'] = i
        robot_config['sigma_initial'] = config['sigma_initial']

        # Make everything a numpy array.
        for key in robot_config:
            if type(robot_config[key]) == list:
                try:
                    # We might be able to make it a numpy array.
                    robot_config[key] = np.array(robot_config[key], dtype=float)
                except:
                    # But if not, leave it as the mixed types array it is.
                    pass

        robots.append(Robot(robot_config))
        motions.append(RobotMotion(robot_config))

    return robots, motions


def dist(p1, p2):
    """Euclidean distance between two points.

    Args:
        p1: numpy vector for n-dimensional position.
        p2: numpy vector for n-dimensional position.

    Returns:
        Euclidean distance between p1, p2
    """

    return np.linalg.norm(p2 - p1)


def main():
    logging.basicConfig(level=logging.DEBUG)
    if not args.random:
        np.random.seed(42)
    
    with open(args.robot_config) as f:
        config = yaml.load(f.read())

    #  robots = initialize_robots(config)
    #  motions = initialize_robot_motions(config)
    robots, motions = initialize_robots_and_motions(config)
    vis = Visualizer(robots, motions)
    num_robots = len(robots)

    while True:
        for i, (robot, motion) in enumerate(zip(robots, motions)):

            # Ground truth for the robots will be stored elsewhere.
            control_output = robot.get_control_output()
            odometry = motion.apply_control_input(control_output)
            robot.receive_odometry_message(odometry)

        for i in range(num_robots):
            for j in range(i+1, num_robots): # No self messages
                robot1 = robots[i]
                robot2 = robots[j]
                motion1 = motions[i]
                motion2 = motions[j]

                # If within long range, exchange long range sensor measurements.
                if dist(motion1.pos, motion2.pos) < config['long_thresh']:
                    # Get both message first.
                    message1to2 = robot1.get_long_range_message()
                    message2to1 = robot2.get_long_range_message()
   
                    # And then transmit both of the messages.
                    robot1.receive_long_range_message(message2to1)
                    robot2.receive_long_range_message(message1to2)

                    # Do it this way to ensure that receiving a message doesn't
                    # modify some state inside the robot.
               
                # If within short range, exchange more detailed data. Note that
                # this allows both long range and short range measurements to be
                # sent when the robots are close enough, with the long range
                # measurements being sent first.
                if dist(motion1.pos, motion2.pos) < config['long_thresh']:
                    # Get both message first.
                    message1to2 = robot1.get_short_range_message()
                    message2to1 = robot2.get_short_range_message()
   
                    # And then transmit both of the messages.
                    robot1.receive_short_range_message(message2to1)
                    robot2.receive_short_range_message(message1to2)

        # Let each robot perform some computation at each time step.
        for robot in robots:
            robot.compute()

        # Perform visualization update.
        vis.update()

        # As the final step of the loop, update the timestamp for each robot.
        for robot in robots:
            robot.step(1)

        time.sleep(1)




if __name__ == "__main__":
    main()