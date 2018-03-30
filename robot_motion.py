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
        self.vel = np.zeros(2)


    def apply_control_input(self, control_input):
        """Applies control input to ground truth state.

        This is a "physics simulator" for the robot. Given the control input,
        the ground truth will update. Based on the update, an odometry reading
        will be returned.

        Returns:
            odometry measurement.
        """

        # Keep the ground truth perfect.
        self.pos += control_input

        # Odometry output has noise associated with it.
        return control_input + np.random.normal(0, 0.1, 2)
