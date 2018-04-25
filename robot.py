import logging
import math
import numpy as np
import scipy.linalg
import os

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
        np.set_printoptions(precision=3, linewidth=os.get_terminal_size().columns)

        self.logger.info("Initializing with config %s", config)
        self.config = config
        self.id = config['id']
        self.goal = config['goal']
        sigma_init = config['sigma_initial']
        self.sigma_init_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_init))
        sigma_odom = config['sigma_odom']
        self.sigma_odom_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
        self.sigma_other_odom_inv = self.sigma_odom_inv
        sigma_range = config['sigma_range']
        self.sigma_range_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_range))
        self.sensor_deltas = [np.array(s['delta']) for s in config['sensor_parameters']]

        self.start_pos = config['start']
        self.odom_measurements = []
        self.range_measurements = []

        #buffer of range measurements received before successfully triangulating
        #an initial pose
        self.initial_ranges = dict()
        #robots we've received new measurements of
        self.update_ids = set()
        self.n_poses = {self.id : 1}
        self.pose_dim = 3
        self.odom_dim = 3
        self.range_dim = 1
        self.fake_thetas = True

        self.x = np.zeros((self.pose_dim, 1))
        self.x[:, 0] = (self.th, self.pos[0], self.pos[1])

        # Motion Controller Params, could be moved into config
        self.kp_pos = .1
        self.kp_th = 1
        self.v_lin_max = 1
        self.v_th_max = math.pi / 128

        self.t = 0

        self.max_iterations = 50
        self.stopping_threshold = 1e-6

    def start_of_next_robot(self,robot_id):
        #count all the poses until "this" robot's is reached.
        #This could require counting all the robots before you (id < self.id)
        #and all of their respective poses (sum())
        start = sum([self.n_poses[id] for id in self.n_poses if id < robot_id])

        #returns the index of the first state element of the first pose after ours
        j0 = start + self.pose_dim * self.n_poses[robot_id]
        return j0

    @property
    def pos(self):
        j0 = self.start_of_next_robot(self.id)
        return self.x[j0 - 2:j0].flatten()

    @property
    def th(self):
        j0 = self.start_of_next_robot(self.id)
        return self.x[j0 - 3]

    def robot_pos(self, id, t = None):
        """Return this robot's belief in the x,y position of the specified robot

        Args:
            id: The id of the robot whose position is being queried
            t: If provided this function will return the belief of the robot's
               position at time t

        Returns:
            The x,y position being queried, or None if the specified robot
            hasn't been seen yet
        """
        if not id in self.n_poses:
          return (None, None)
        j0 = self.start_of_next_robot(id)
        if self.n_poses[id] == 0:
            return np.array([None, None])
        return self.x[j0 - 2:j0].flatten()


    def robot_th(self, id, t):
        """Return this robot's belief in the angle of the specified robot

        Args:
            id: The id of the robot whose angle is being queried
            t: If provided this function will return the belief of the robot's
               angle at time t

        Returns:
            The angle being queried
        """
        j0 = self.start_of_next_robot(id)
        return self.x[j0 - 3]


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
            #don't have an initial estimate yet, so buffer measurements until we
            #can successfully triangulate
            if not other_id in self.initial_ranges.keys():
              self.initial_ranges[other_id] = []
            for i, r in enumerate(message.measurements):
              self.initial_ranges[other_id].append((ind, i, r))
        elif False:
            self.update_ids.add(other_id)
            other_ind = self.n_poses[other_id]
            for i, r in enumerate(message.measurements):
                #measurements stored as (self_pose_index, other_id,
                #                        other_pose_index, sensor_index, range)
                meas = (ind, other_id, other_ind, i, r)
                self.range_measurements.append(meas)


    def receive_odometry_message(self, message):
        """Receive an odometry message to be able to sense the motion.

        This odometry measurement should be a result of the control output.
        Odometry Format:  [th_dot, x_dot, y_dot]

        Args:
            message: Odometry measurement as a result of control output.
        """
        self.logger.debug("Received odometry message %s", message)
        self.odom_measurements.append(np.expand_dims(message, axis=1))


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

        #Use proportional controllers on linear and angular velocity
        scale = self.v_lin_max / np.abs(d_pos).max()
        v_lin = d_pos if scale > 1 else d_pos * scale
        self.control_output = np.hstack(([0], v_lin))
        self.logger.debug("Returning control output %s", self.control_output)
        return self.control_output

    def iterative_update(self,A,b):
        A_sp = csc_matrix(A.T.dot(A))
        A_splu = splu(A_sp)
        prev_x = self.x
        dx = A_splu.solve(A.T.dot(b))
        self.x = prev_x + dx

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
        for i, m in enumerate(self.odom_measurements):
            start = sum([self.n_poses[id] for id in self.n_poses if id < self.id])
            j0 = self.pose_dim * (i + start)
            j1 = self.pose_dim * (i + 1 + start)
            p0 = self.x[j0:j0 + self.pose_dim]
            p1 = self.x[j1:j1 + self.pose_dim]
            H = np.array([[-1, 0, 0, 1, 0, 0],
                          [0, -1, 0, 0, 1, 0],
                          [0, 0, -1, 0, 0, 1]])
            H = self.sigma_odom_inv.dot(H)
            A[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
            A[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
            #TODO: odom measurements should be multiplied by dt
            mPred = p1 - p0
            mPred[0] = self.wrapToPi(mPred[0])
            dz = m - mPred
            dz[0] = self.wrapToPi(dz[0])
            b[i0 + self.odom_dim*i:i0 + self.odom_dim*(i + 1), 0] = dz.flatten()
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
            start = sum([self.n_poses[id] for id in self.n_poses if id < self.id])
            other_start = sum([self.n_poses[id] for id in self.n_poses if id < other_id])
            j0 = self.pose_dim * (ind + start)
            j1 = self.pose_dim * (other_ind + other_start)
            p0 = self.x[j0:j0 + self.pose_dim]
            p1 = self.x[j1:j1 + self.pose_dim]
            th = p0[0, 0]
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th),  np.cos(th)]])

            delta = self.sensor_deltas[si]
            d = p0[1:, 0] + R.dot(delta) - p1[1:, 0]
            r = np.linalg.norm(d)
            H = np.array([[(d[0] * (-delta[0]*np.sin(th) - delta[1]*np.cos(th)) +
                            d[1] * (delta[0]*np.cos(th) - delta[1]*np.sin(th))) / r,
                           d[0] / r, d[1] / r,  0, -d[0] / r, -d[1] / r]])
            H = self.sigma_range_inv.dot(H)
            A[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), j0:j0 + self.pose_dim] = H[0, :self.pose_dim]
            A[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), j1:j1 + self.pose_dim] = H[0, self.pose_dim:]
            b[i0 + self.range_dim*i:i0 + self.range_dim*(i+1), 0] = rMeas - r
        return self.range_dim * len(self.range_measurements)


    def build_other_odom_system(self, A, b, i0):
        """builds the block of A and b corresponding to odometry measurements
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Number of rows in this block
        """
        i = i0
        for id in self.n_poses:
            if id == self.id:
                continue
            start = sum([self.n_poses[_id] for _id in self.n_poses if _id < id])
            for n in range(self.n_poses[id] - 1):
                j0 = self.pose_dim * (start + n)
                j1 = self.pose_dim * (start + n + 1)
                p0 = self.x[j0:j0 + self.pose_dim]
                p1 = self.x[j1:j1 + self.pose_dim]
                H = np.array([[-1, 0, 0, 1, 0, 0],
                              [0, -1, 0, 0, 1, 0],
                              [0, 0, -1, 0, 0, 1]])
                H = self.sigma_other_odom_inv.dot(H)
                A[i:i + self.odom_dim, j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
                A[i:i + self.odom_dim, j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
                mPred = p1 - p0
                mPred[0] = self.wrapToPi(mPred[0])
                dz = -mPred
                dz[0] = self.wrapToPi(dz[0])
                b[i:i + self.odom_dim, 0] = dz.flatten()
                i += self.odom_dim
        return i - i0


    def build_theta_priors(self, A, b, i0):
        """Put priors on the orientations of other robots so the system is fully
        constrained. Should only use when sensor measurements are insufficient
        to figure out the orientation
        """
        i = i0
        for id in self.n_poses:
            if id == self.id:
                continue
            start = sum([self.n_poses[_id] for _id in self.n_poses if _id < id])
            j0 = self.pose_dim * start
            A[i, j0] = 1
            i += 1
        return i - i0


    def build_system(self):
        """Build A and b linearized around the current state
        """
        M = len(self.odom_measurements) * self.odom_dim + self.pose_dim + \
            len(self.range_measurements) * self.range_dim + \
            (sum(self.n_poses.values()) - self.n_poses[self.id] - len(self.n_poses) + 1) * self.odom_dim
        if self.fake_thetas:
            M += len(self.n_poses) - 1 # -1 for this robot
        N = sum(self.n_poses.values()) * self.pose_dim
        A = scipy.sparse.lil_matrix((M, N))
        b = np.zeros((M, 1))

        #Prior
        A[:self.pose_dim, :self.pose_dim] = self.sigma_init_inv
        b[:self.pose_dim, 0] = self.start_pos

        i0 = self.pose_dim
        if self.fake_thetas:
          i0 += self.build_theta_priors(A, b, i0)
        i0 += self.build_odom_system(A, b, i0)
        i0 += self.build_other_odom_system(A, b, i0)
        i0 += self.build_range_system(A, b, i0)
        #print(A.toarray())

        return A, b

    def triangulate(self, measurements):
        """Triangulate position of other robot based on sensor readings

        This method is used to estimate an initial position for another robot
        the first time it is seen
        """
        if len(measurements) < 3:
          return None
        start = sum([self.n_poses[id] for id in self.n_poses if id < self.id])
        A = np.zeros((len(measurements) - 1, 2))
        b = np.zeros((len(measurements) - 1, 1))

        #Find x_n, y_n, r_n
        ind_n, si_n, r_n = measurements[-1]
        delta = self.sensor_deltas[si_n]
        th_n = self.x[start + self.pose_dim * ind_n, 0]
        R = np.array([[np.cos(th_n), -np.sin(th_n)],
                      [np.sin(th_n),  np.cos(th_n)]])
        d = R.dot(delta)
        x_n = self.x[start + self.pose_dim * ind_n + 1] + d[0]
        y_n = self.x[start + self.pose_dim * ind_n + 2] + d[1]

        for i, (ind, si, r) in enumerate(measurements[:-1]):
            delta = self.sensor_deltas[si]
            th = self.x[start + self.pose_dim * ind, 0]
            R = np.array([[np.cos(th), -np.sin(th)],
                          [np.sin(th),  np.cos(th)]])
            d = R.dot(delta)
            x = self.x[start + self.pose_dim * ind + 1] + d[0]
            y = self.x[start + self.pose_dim * ind + 2] + d[1]
            A[i, :] = (x_n - x, y_n - y)
            b[i] = r**2 - r_n**2 - x**2 + x_n**2 - y**2 + y_n**2

        return np.linalg.lstsq(A, b)[0]

    def compute(self):
        """Perform all the computation required to process messages.

        This method is called every time step, before time is updated in the
        robot (before .step()). This method should be the one the robot uses to
        perform all of the SLAM updates.
        """
        self.logger.debug("Computing at time %d", self.t)
        #use previous pose as current pose estimate

        j0 = self.start_of_next_robot(self.id)
        p_new = self.x[j0-self.pose_dim:j0]
        self.x = np.insert(self.x, j0, p_new, axis=0)
        self.n_poses[self.id] += 1

        done_triangulating = []
        for other_id in self.initial_ranges.keys():
            pos = self.triangulate(self.initial_ranges[other_id])
            if pos is not None: #Triangulation was successful
                done_triangulating.append(other_id)
                #Use pos as initial state estimate (set th = 0)
                start = sum([self.n_poses[id] for id in self.n_poses if id < other_id])
                self.x = np.insert(self.x, start, np.vstack((pos, [0])), axis=0)
                self.n_poses[other_id] = 1

                #Use the most recent measurement for SLAM
                ind = self.n_poses[self.id] - 1
                measurements = [(i, r) for ind0, i, r in self.initial_ranges[other_id] if ind0 == ind]
                for i, r in measurements:
                    #measurements stored as (self_pose_index, other_id,
                    #                        other_pose_index, sensor_index,
                    #                        range)
                    meas = (ind, other_id, 0, i, r)
                    self.range_measurements.append(meas)
        for id in done_triangulating:
            del self.initial_ranges[id]

        for other_id in self.update_ids:
            j0 = self.start_of_next_robot(other_id)
            p_new = self.x[j0-self.pose_dim:j0]
            self.x = np.insert(self.x, j0, p_new, axis=0)
            self.n_poses[other_id] += 1

        for i in range(0,self.max_iterations):
            A,b = self.build_system()
            if(not self.iterative_update(A,b)):
                break

        if self.id == 1:
          print(self.x)
        #reset update_ids
        self.update_ids = set()


    def step(self, step):
        """Increment the robot's internal time state by some step size.

        Args:
            step: time to be added to internal state
        """
        self.t += step
        self.logger.debug("Adding time %d to get new time %d", step, self.t)
