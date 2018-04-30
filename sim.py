#!/usr/bin/env python3

import argparse
import math
import numpy as np
import logging
import multiprocessing
import sys
import time
import yaml

from robot import Robot
from robot_motion import RobotMotion
from vis import Visualizer
from long_range_message import LongRangeMessage
from sensor_model import SensorModel

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
parser.add_argument(
    "-d", "--delay",
    type=float,
    default=0.5,
    help="Time to sleep (in seconds) between time steps."
)
parser.add_argument(
    "-x", "--headless",
    action="store_true",
    default=False,
    help="Disable visualization (and movie writing)."
)
parser.add_argument(
    "-t", "--time",
    type=int,
    default=25,
    help="Number of iterations to run the simulator for before exiting. "
         " A time of 0 will result in an infinite simulation."
)
parser.add_argument(
    "-m", "--multiprocessing",
    action="store_true",
    default=False,
    help="Use multiprocessing during the compute step."
)
parser.add_argument(
    "-l", "--loglevel",
    type=lambda s: s.upper(),
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
    default='DEBUG',
    help="Set the logging level"
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
        robot_config['sensor_parameters'] = config['sensor_parameters']
        robot_config['goal_parameters'] = config['goal_parameters']
        motion_config = robot_config.copy()
        robot_config.update(config['slam_parameters'])
        motion_config.update(config['motion_parameters'])

        # Make everything a numpy array.
        for key in robot_config:
            if type(robot_config[key]) == list:
                try:
                    # We might be able to make it a numpy array.
                    robot_config[key] = np.array(robot_config[key], dtype=float)
                except:
                    # But if not, leave it as the mixed types array it is.
                    pass

        # Make everything a numpy array.
        for key in motion_config:
            if type(motion_config[key]) == list:
                try:
                    # We might be able to make it a numpy array.
                    motion_config[key] = np.array(motion_config[key], dtype=float)
                except:
                    # But if not, leave it as the mixed types array it is.
                    pass


        robots.append(Robot(robot_config))
        motions.append(RobotMotion(motion_config))

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


def do_long_range_message(config, robot1, robot2, motion1, motion2):
    """Potentially transfer long range messages between robots.

    Depending on whether the robots are within thresh distance, they will
    exchange long range measurements. The long range measurements are sent in
    the form of LongRangeMeasurement objects.

    Args:
        config: Configuration file.
        robot1: First Robot object.
        robot2: Second Robot object.
        motion1: First RobotMotion object.
        motion2: Second RobotMotion object.
    """

    # If within long range, exchange long range sensor measurements.
    if dist(motion1.pos, motion2.pos) < config['long_thresh']:
        # Get message data from each robot.
        message1to2_data = robot1.get_long_range_message()
        message2to1_data = robot2.get_long_range_message()

        # Create the sensor model, which will be used to make all measurements.
        sm = SensorModel(config)

        # Compute the pairwise sensor measurements between each robot.
        num_sensors = len(config['sensor_parameters'])

        robot1_measurements = [sm.get_measurement(motion1, i, motion2)
                for i in range(num_sensors)]
        robot2_measurements = [sm.get_measurement(motion2, i, motion1)
                for i in range(num_sensors)]

        logging.debug("Robot 1 Measurements: %s", robot1_measurements)
        logging.debug("Robot 2 Measurements: %s", robot2_measurements)

        # Stuff the data into the LongRangeMessages
        message1to2 = LongRangeMessage(message1to2_data, robot2_measurements)
        message2to1 = LongRangeMessage(message2to1_data, robot1_measurements)

        # And then transmit both of the messages.
        # Do it this way to ensure that receiving a message doesn't
        # modify some state inside the robot.
        robot1.receive_long_range_message(message2to1)
        robot2.receive_long_range_message(message1to2)

        # indicate whether these two robots communicated
        return True
    return False


def do_short_range_message(config, robot1, robot2, motion1, motion2):
    """Potentially transfer long range messages between robots.

    Depending on whether the robots are within thresh distance, they will
    exchange short range measurements. The data exchanged is just what is
    returned by get_short_range_message(), with no additional sensor data.

    Args:
        config: Configuration file.
        robot1: First Robot object.
        robot2: Second Robot object.
        motion1: First RobotMotion object.
        motion2: Second RobotMotion object.
    """
    # If within short range, exchange more detailed data. Note that
    # this allows both long range and short range measurements to be
    # sent when the robots are close enough, with the long range
    # measurements being sent first.
    if dist(motion1.pos, motion2.pos) < config['short_thresh']:
        # Get both message first.
        message1to2 = robot1.get_short_range_message()
        message2to1 = robot2.get_short_range_message()

        # And then transmit both of the messages.
        robot1.receive_short_range_message(message2to1)
        robot2.receive_short_range_message(message1to2)

        # indicate whether the two robots communicated
        return True
    return False


def compute_map(robot):
    robot.compute()
    return robot


def main():
    logging.basicConfig(level=args.loglevel)
    if not args.random:
        np.random.seed(42)

    with open(args.robot_config) as f:
        config = yaml.load(f.read())

    robots, motions = initialize_robots_and_motions(config)
    num_robots = len(robots)

    if not args.headless:
        vis = Visualizer(num_robots, config['motion_parameters']['disturbance'])

    pool = multiprocessing.Pool()

    elapsed = 0
    while True:
        # create array of pairs of robots that sent messages
        long_range_measurements = []
        short_range_measurements = []

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

                # Potentially exchange long / short messages, and log their results
                did_short_range_comm = do_long_range_message(
                        config, robot1, robot2, motion1, motion2)
                did_long_range_comm = do_short_range_message(
                        config, robot1, robot2, motion1, motion2)

                # if we exchanged the messages, add the indices of the pair to the visualizer
                if did_short_range_comm:
                    short_range_measurements.append((i, j))
                if did_long_range_comm:
                    long_range_measurements.append((i, j))

        # Let each robot perform some computation at each time step.
        start_time = time.time()
        if args.multiprocessing:
            robots = pool.map(compute_map, robots)
        else:
            for robot in robots:
                robot.compute()
        end_time = time.time()
        logging.warning("Took %f seconds to perform all computation!",
                        end_time - start_time)

        # Perform visualization update.
        if not args.headless:
            vis.update(robots, motions, short_range_measurements, 
                       long_range_measurements)

        # As the final step of the loop, update the timestamp for each robot.
        for robot in robots:
            robot.step(1)

        # Potentially exit the simulation.
        elapsed += 1
        if args.time > 0 and elapsed >= args.time:
            sys.exit(0)

        # Sleep for some amount of time.
        if (args.delay > 0):
            time.sleep(args.delay)
        elif (args.delay < 0):
            input()


if __name__ == "__main__":
    main()
