import logging
import math
import numpy as np
import scipy.linalg

from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.spatial.distance import euclidean

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
        self.sensor_deltas = [np.array(s['delta']) for s in config['sensor_parameters']]

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

        self.max_iterations = 50
        self.stopping_threshold = 1e-6


    def wrapToPi(self, angle):
        """Wrap a measurement in radians to [-pi, pi]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi


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
        other_id = message.data['id']
        ind = self.n_poses[self.id]
        if not other_id in self.n_poses:
          self.n_poses[other_id] = 0
        otherInd = self.n_poses[other_id]
        for i, r in enumerate(message.measurements):
          #measurements stored as (self_pose_index, other_id, other_pose_index,
          #                        sensor_index, range)
          self.range_measurements.append((ind, other_id, otherInd, i, r))


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

    def iterative_update(self,A,b):
        A_sp = csc_matrix(A.T.dot(A))
        A_splu = splu(A_sp)
        prev_x = self.x
        self.x = A_splu.solve(A.T.dot(b))

        if(euclidean(self.x.T,prev_x) < self.stopping_threshold):
            return False
        return True

    def build_odom_system(self, A, b, i0):
        """builds the block of A and b corresponding to odometry measurements
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Number of rows in this block
        """
        for i, (v, w) in enumerate(self.odom_measurements):
            start = sum([self.n_poses[id] for id in self.n_poses if id < self.id])
            j0 = self.pose_dim * (i + start)
            j1 = self.pose_dim * (i + 1 + start)
            p0 = self.x[j0:j0 + self.pose_dim]
            p1 = self.x[j1:j1 + self.pose_dim]
            eps = 1e-7
            th = math.atan2(p1[2] - p0[2] + eps, p1[1] - p0[1] + eps)
            a0 = self.wrapToPi(th - p0[0])
            l = math.sqrt((p1[1] - p0[1] + eps)**2 + (p1[2] - p0[2] + eps)**2)
            a1 = self.wrapToPi(p1[0] - th)
            H = np.array([[-1, (p0[2] - p1[2] + eps) / l**2, (p0[1] - p1[1] + eps) / l**2,
                           0, (p1[2] - p0[2] + eps) / l**2, (p1[1] - p0[1] + eps) / l**2],
                          [0, (p0[1] - p1[1] + eps) / l, (p0[2] - p1[2] + eps) / l,
                           0, (p1[1] - p0[1] + eps) / l, (p1[2] - p0[2] + eps) / l],
                          [0, (p1[2] - p0[2] + eps) / l**2, (p1[1] - p0[1] + eps) / l**2,
                           1, (p0[2] - p1[2] + eps) / l**2, (p0[1] - p1[1] + eps) / l**2]])
            H = self.sigma_odom_inv.dot(H)
            A[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
            A[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
            #TODO: odom measurements should be multiplied by dt
            b[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), 0] = [self.wrapToPi(0 - a0), v - l, self.wrapToPi(w - a1)]
        return self.odom_dim * len(self.odom_measurements)


    def build_range_system(self, A, b, i0):
        """Builds the block of A and b corresponding to range measurements
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Number of rows in this block

        """
        for i, (ind, other_id, other_ind, si, rMeas) in enumerate(self.range_measurements):
            start = sum([self.n_poses[id] for id in self.poses if id < self.id])
            other_start = sum([self.n_poses[id] for id in self.poses if id < other_id])
            j0 = self.pose_dim * (ind + start)
            j1 = self.pose_dim * (other_ind + other_start)
            p0 = self.x[j0:j0 + self.pose_dim]
            p1 = self.x[j1:j1 + self.pose_dim]
            th = p0[0]
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th),  np.cos(th)]])

            delta = self.sensor_deltas[si]
            d = p0[1:] + R.dot(delta) - p0[1:]
            r = np.linalg.norm(d)
            H = np.array([[(d[0] * (-delta[0]*np.sin(th) - delta[1]*np.cos(th)) +
                            d[1] * (delta[0]*np.cos(th) - delta[1]*np.sin(th))) / r,
                           d[0] / r, d[1] / r,  0, -d[0] / r, -d[1] / r]])
            H = self.sigma_range_inv.dot(H)
            A[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
            A[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
            b[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), 0] = rMeas - r


    def build_system(self):
        """Build A and b linearized around the current state
        """
        M = len(self.odom_measurements) * self.odom_dim + \
            self.pose_dim
            #len(self.range_measurements) * self.range_dim + \
        N = sum(self.n_poses.values()) * self.pose_dim
        A = scipy.sparse.lil_matrix((M, N))
        b = np.zeros((M, 1))

        #Prior
        A[:self.pose_dim, :self.pose_dim] = self.sigma_init_inv
        b[:self.pose_dim, 0] = self.start_pos

        i0 = self.pose_dim
        i0 += self.build_odom_system(A, b, i0)
        #i0 += self.build_range_system(A, b, i0)

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

        dir = np.array([math.cos(self.th), math.sin(self.th)])
        self.pos += self.odom_measurements[-1][0] * dir
        self.th += self.odom_measurements[-1][1]

#        for i in range(0,self.max_iterations):
#            A,b = self.build_system()
#            if(not self.iterative_update(A,b)):
#                break
#
#        start = sum([self.n_poses[id] for id in self.n_poses if id < self.id])
#        j0 = start + self.pose_dim * (self.n_poses[self.id] - 1)
#        self.pos = self.x[j0 + 1:j0 + 3].flatten()
#        self.th = self.x[j0]

    def step(self, step):
        """Increment the robot's internal time state by some step size.

        Args:
            step: time to be added to internal state
        """
        self.t += step
        self.logger.debug("Adding time %d to get new time %d", step, self.t)
