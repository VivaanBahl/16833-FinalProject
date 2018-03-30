#!/usr/bin/env python3

import argparse
import numpy as np
import logging
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
    "-n", "--num_robots",
    type=int,
    default=3,
    help="Number of robots to simulate."
)
parser.add_argument(
    "-r", "--random",
    action="store_true",
    default=False,
    help="Use random number generation rather than a fixed seed."
)
args = parser.parse_args()


def initialize_robots(config):
    """Initialize the set of robots from the user input.

    Args:
        config: Configuration file.

    Returns:
        List of initialized Robots.
    """

    robots = []
    for i in range(args.num_robots):
        config['id'] = i
        robots.append(Robot(config.copy()))

    return robots


def initialize_robot_motions(config):
    """Initialize robot motions corresponding to each robot.

    The idea is to create a software separation between the robot and its ground
    truth. This is so that we don't accidentally use ground truth data in the
    robot, forcing all of our code to be correct.

    Args:
        config: Configuration file.

    Returns:
        list of initialized RobotMotions.
    """
    motions = [RobotMotion() for i in range(args.num_robots)]
    return motions


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

    robots = initialize_robots(config)
    motions = initialize_robot_motions(config)
    vis = Visualizer(robots, motions)

    while True:
        for i, (robot, motion) in enumerate(zip(robots, motions)):

            # Ground truth for the robots will be stored elsewhere.
            control_output = robot.get_control_output()
            odometry = motion.apply_control_input(control_output)
            robot.receive_odometry_message(odometry)

        for i in range(args.num_robots):
            for j in range(i+1, args.num_robots): # No self messages
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


        # Perform visualization update.
        vis.update()

        # As the final step of the loop, update the timestamp for each robot.
        for robot in robots:
            robot.step(1)




if __name__ == "__main__":
    main()
