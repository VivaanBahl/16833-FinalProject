import math
import numpy as np


class RobotMotion(object):
    """A Ground Truth System for a Robot.

    This object manages the ground truth of a Robot, while also serving as the
    motion model and odometry sensor.
    """

    def __init__(self):
        """Initialize robot motion.

        Stores a position and velocity in 2D.
        """
        self.pos = np.zeros(2)
        self.th = 0
        self.vel = np.zeros(2)


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
