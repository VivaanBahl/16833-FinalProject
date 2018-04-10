import logging
import math
import numpy as np
import scipy.linalg

class Robot(object):
    def __init__(self, config):
        """Initialize robot.

        Intializes robot from configuration file. The start position and
        covariance are specified from the configuration, while the ground truth
        is stored in the corresponding RobotMotion.

        Args:
            config: Configuration file.
        """
        self.logger = logging.getLogger("Robot %d" % config['id'])

        self.logger.info("Initializing with config %s", config)
        self.config = config
        self.id = config['id']
        self.goal = config['goal']
        self.pos = config['start'][1:]
        self.th = config['start'][0]
        sigma_init = config['sigma_initial']
        self.sigma_init_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_init))
        sigma_odom = config['sigma_odom']
        self.sigma_odom_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))

        self.start_pos = config['start']
        self.odom_measurements = []
        self.range_measurements = []
        self.n_poses = {self.id : 1}
        self.pose_dim = 3
        self.odom_dim = 3
        self.range_dim = 1

        self.x = np.zeros((self.pose_dim, 1))
        self.x[:, 0] = (self.th, self.pos[0], self.pos[1])

        # Motion Controller Params, could be moved into config
        self.kp_pos = .1
        self.kp_th = 1
        self.v_lin_max = 10
        self.v_th_max = math.pi / 16

        self.t = 0

    def receive_short_range_message(self, message):
        """Handle a short range message.

        Short range messages should be high bandwidth, for when this robot is
        within a certain range of any other robot.

        Args:
            message: An object returned by get_short_range_data() call from
                another robot.
        """
        self.logger.debug("Received short range message %s", message)


    def receive_long_range_message(self, message):
        """Handle a long range message.

        Long range messages should be low bandwidth, for when this robot is far
        away from other robots but still within long range sensor range.
        """
        self.logger.debug("Received long range message %s", message)


    def receive_odometry_message(self, message):
        """Receive an odometry message to be able to sense the motion.

        This odometry measurement should be a result of the control output.
        Odometry Format:  [linear_velocity, angular_velocity]

        Args:
            message: Odometry measurement as a result of control output.
        """
        self.logger.debug("Received odometry message %s", message)
        self.odom_measurements.append(message)


    def get_short_range_message(self):
        """Get short range message to transmit to another robot.

        This message is allowed to be large, but should only be transmitted when
        the distance between robots is under a certain amount.

        Returns:
            Object to be transmitted to other robots when in short range.
        """

        message = {"id": self.config['id']}
        message["data"] = [2]
        self.logger.debug("Returning short range message %s", message)
        return message


    def get_long_range_message(self):
        """Get long range message to transmit to another robot.

        This message is to be transmitted along with measurements for long
        range communication. The objects returned from here will be relatively
        small.

        Returns:
            Object to be transmitted to other robots in long range.
        """

        message = {"id": self.config['id']}
        message["data"] = [42]
        self.logger.debug("Returning long range message %s", message)
        return message


    def get_control_output(self):
        """Get robot's control output for the current state.

        The robot may attempt to effect some sort of motion at every time step,
        which will be a result of the control output from this time step.

        Control Input:  [linear_velocity, angular_velocity]

        Returns:
            Control output to feed into the robot model.
        """

        #compute position and orientation errors relative to goal
        d_pos = self.goal - self.pos
        goal_th = math.atan2(d_pos[1], d_pos[0])
        d_th = (goal_th - self.th + math.pi) % (2*math.pi) - math.pi

        #Use proportional controllers on linear and angular velocity
        v_lin = min(self.kp_pos * np.linalg.norm(d_pos), self.v_lin_max)
        v_th = min(self.kp_th * d_th, self.v_th_max)
        control_output = np.array([v_lin, v_th])
        self.logger.debug("Returning control output %s", control_output)
        return control_output

    def build_system(self):
        """Build A and b linearized around the current state
        """
        M = len(self.odom_measurements) * self.odom_dim + \
            len(self.range_measurements) * self.range_dim + \
            self.pose_dim
        N = sum(self.n_poses.values()) * self.pose_dim
        A = scipy.sparse.lil_matrix((M, N))
        b = np.zeros((M, 1))

        #Prior
        A[:self.pose_dim, :self.pose_dim] = self.sigma_init_inv
        b[:self.pose_dim, 0] = self.start_pos

        #Odometry Measurements
        for i, (v, w) in enumerate(self.odom_measurements):
          i0 = self.pose_dim * i
          i1 = self.pose_dim * (i + 1)
          p0 = self.x[i0:i0 + self.pose_dim]
          p1 = self.x[i1:i1 + self.pose_dim]
          l = math.sqrt((p1[1] - p0[1])**2 + (p1[2] - p0[2])**2)
          H = np.array([[-1, (p0[2] - p1[2]) / l**2, (p0[1] - p1[1]) / l**2,
                         0, (p1[2] - p0[2]) / l**2, (p1[1] - p0[1]) / l**2],
                        [0, (p0[1] - p1[1]) / l, (p0[2] - p1[2]) / l,
                         0, (p1[1] - p0[1]) / l, (p1[2] - p0[2])],
                        [0, (p1[2] - p0[2]) / l**2, (p1[1] - p0[1]) / l**2,
                         -1, (p0[2] - p1[2]) / l**2, (p0[1] - p1[1]) / l**2]])
          H = self.sigma_odom_inv.dot(H)
          A[self.odom_dim*i + 1:self.odom_dim*i + self.odom_dim + 1, i0:i0 + self.pose_dim] = H[:, :self.pose_dim]
          A[self.odom_dim*i + 1:self.odom_dim*i + self.odom_dim + 1, i1:i1 + self.pose_dim] = H[:, self.pose_dim:]
          b[self.odom_dim*i + 1:self.odom_dim*i + self.odom_dim + 1, 0] = [0, v, w]
        return A, b


    def compute(self):
        """Perform all the computation required to process messages.

        This method is called every time step, before time is updated in the
        robot (before .step()). This method should be the one the robot uses to
        perform all of the SLAM updates.
        """
        self.logger.debug("Computing at time %d", self.t)
        #use previous pose as current pose estimate
        self.x = np.vstack((self.x, self.x[-self.pose_dim:]))
        self.n_poses[self.id] += 1
        self.build_system()

        dir = np.array([math.cos(self.th), math.sin(self.th)])
        self.pos += self.odom_measurements[-1][0] * dir
        self.th += self.odom_measurements[-1][1]

    def step(self, step):
        """Increment the robot's internal time state by some step size.

        Args:
            step: time to be added to internal state
        """
        self.t += step
        self.logger.debug("Adding time %d to get new time %d", step, self.t)
