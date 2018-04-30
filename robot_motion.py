import math
import logging
import numpy as np
import disturbances as db

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
        x0 = np.random.multivariate_normal(
                config['start'],
                config['sigma_initial']
        )
        self.sigma = config['sigma_odom']
        self.sigma_control = config['sigma_control']
        if config['disturbance'] == 'radial_waves':
          self.disturbance = db.radial_waves
        if config['disturbance'] == 'linear':
          self.disturbance = db.linear
        else:
          self.disturbance = db.no_force
        self.pos = x0[1:]
        self.th = (x0[0] + math.pi) % (2*math.pi) - math.pi
        self.vel = np.zeros(len(self.pos))

        self.logger.debug("Initialized with position %s", self.pos)


    def apply_control_input(self, control_input):
        """Applies control input to ground truth state.

        This is a "physics simulator" for the robot. Given the control input,
        the ground truth will update. Based on the update, an odometry reading
        will be returned.

        Control Input/Odometry Format:  [x_dot, y_dot, th_dot]

        Returns:
            odometry measurement.
        """

        # Keep the ground truth perfect.

        noisy_control = np.random.multivariate_normal(control_input, self.sigma_control)

        self.vel = noisy_control[1:]

        f = self.disturbance(self.th,self.pos[0],self.pos[1])

        d_th = noisy_control[0] + f[0]
        self.th += d_th
        self.th = (self.th + math.pi) % (2*math.pi) - math.pi

        d_pos = self.vel + f[1:2]
        self.pos += d_pos

        # Odometry output has noise associated with it.
        return np.random.multivariate_normal([d_th, d_pos[0], d_pos[1]], self.sigma)
