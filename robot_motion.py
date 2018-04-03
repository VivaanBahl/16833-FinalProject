import math
import logging
import numpy as np


class RobotMotion(object):
    """A Ground Truth System for a Robot.

    This object manages the ground truth of a Robot, while also serving as the
    motion model and odometry sensor.
    """

    def __init__(self, config):
        """Initialize robot motion.

        Stores a position and velocity in 2D. The ground truth for the robot is
        sampled from the initial sigma specified in the configuration.
        """
        self.logger = logging.getLogger("Robot Motion %d" % config['id'])

        self.pos = np.random.multivariate_normal(
                config['start'],
                config['sigma_initial']
        )
        self.th = 0
        self.vel = np.zeros(len(self.pos))

        self.logger.debug("Initialized with position %s", self.pos)


    def apply_control_input(self, control_input):
        """Applies control input to ground truth state.

        This is a "physics simulator" for the robot. Given the control input,
        the ground truth will update. Based on the update, an odometry reading
        will be returned.

        Control Input/Odometry Format:  [linear_velocity, angular_velocity]

        Returns:
            odometry measurement.
        """

        # Keep the ground truth perfect.
        dir = np.array([math.cos(self.th), math.sin(self.th)])
        self.vel = control_input[0] * dir
        self.pos += self.vel
        self.th += control_input[1]
        self.th = (self.th + math.pi) % (2*math.pi) - math.pi

        # Odometry output has noise associated with it.
        return control_input + np.random.normal(0, [0.1, .001])
