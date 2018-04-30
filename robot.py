import logging
import math
import numpy as np
import scipy.linalg
import shutil

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
        self.logger_name = "Robot %d" % config['id']
        logger = logging.getLogger(self.logger_name)
        np.set_printoptions(precision=3,
                linewidth=shutil.get_terminal_size((80, 24)).columns)

        logger.info("Initializing with config %s", config)

        #initialize settings from config
        self.config = config
        self.id = config['id']

        # Goals initialization
        self.my_goals = config['goals']
        self.goal_index = 0
        self.goal_thresh = config['goal_parameters']['threshold']
        self.loiter_time = config['goal_parameters']['loiter_time']
        self.at_goal_time = 0

        # Covariances initialization
        sigma_init = config['sigma_initial']
        sigma_odom = config['sigma_odom']
        sigma_other_odom = config['sigma_fake_odom']
        sigma_range = config['sigma_range']
        self.sigma_init_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_init))
        self.sigma_odom_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_odom))
        self.sigma_other_odom_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_other_odom))
        self.sigma_range_inv = np.linalg.inv(scipy.linalg.sqrtm(sigma_range))

        self.sensor_deltas = [np.array(s['delta'], dtype=float)
                              for s in config['sensor_parameters']]
        self.use_range = config['use_range']
        self.start_pos = config['start']
        self.short_range_history = config['short_range_history']

        # Motion Controller Params
        self.kp_pos = .1
        self.v_lin_max = 1

        # SLAM Solver Params
        self.max_iterations = 50
        self.stopping_threshold = 1e-6

        self.odom_measurements = []
        self.other_control = {}
        self.range_measurements = []
        self.goals = dict() # Storing goal information for other robot IDs

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
        self.x[:, 0] = self.start_pos
        self.t = 0


    def first_pose_ind(self, robot_id):
        start = sum([self.n_poses[id] for id in self.n_poses if id < robot_id])
        return self.pose_dim * start


    def last_pose_ind(self, robot_id):
        #count all the poses until "this" robot's is reached.
        #This could require counting all the robots before you (id < self.id)
        #and all of their respective poses (sum())
        start = sum([self.n_poses[id] for id in self.n_poses if id < robot_id])

        #returns the index of the first state element of the last pose for this
        #robot
        j0 = self.pose_dim * (start + self.n_poses[robot_id] - 1)
        return j0

    def sensor_pose(self, ind, si, x = None):
        if x is None:
            x = self.x
        delta = self.sensor_deltas[si]
        j = self.first_pose_ind(self.id) + self.pose_dim * ind
        pos = x[j+1:j+3, 0]
        th = x[j, 0]
        R = np.array([[np.cos(th), -np.sin(th)],
                      [np.sin(th),  np.cos(th)]])
        return pos + R.dot(delta)


    @property
    def pos(self):
        j0 = self.last_pose_ind(self.id)
        return self.x[j0 + 1:j0 + 3].flatten()


    @property
    def th(self):
        j0 = self.last_pose_ind(self.id)
        return self.x[j0]


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
        j0 = self.last_pose_ind(id)
        if self.n_poses[id] == 0:
            return np.array([None, None])
        return self.x[j0 + 1:j0 + 3].flatten()


    def robot_th(self, id, t = None):
        """Return this robot's belief in the angle of the specified robot

        Args:
            id: The id of the robot whose angle is being queried
            t: If provided this function will return the belief of the robot's
               angle at time t

        Returns:
            The angle being queried, or None if the specified robot hasn't been
            seen yet
        """
        if not id in self.n_poses:
            return None
        j0 = self.last_pose_ind(id)
        return self.x[j0]


    def wrap_to_pi(self, angle):
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
        logger = logging.getLogger(self.logger_name)
        logger.debug("Received short range message %s", message)
        #ignore message if in odom-only mode
        if not self.use_range:
            return
        #overwrite controls with odometry from short range message
        id = message['id']
        n = len(message['data'])
        self.other_control[id][-n:] = message['data']

    def receive_long_range_message(self, message):
        """Handle a long range message.

        Long range messages should be low bandwidth, for when this robot is far
        away from other robots but still within long range sensor range.
        """
        logger = logging.getLogger(self.logger_name)
        logger.debug("Received long range message %s", message)
        #ignore message if in odom-only mode
        if not self.use_range:
            return
        other_id = message.data['id']
        ind = self.n_poses[self.id]

        # Store the goal information received in the message, overriding
        # previous goal information (if any). This assumes all frames are
        # aligned, which they are.
        self.goals[other_id] = message.data['goal']

        if not other_id in self.n_poses:
            #don't have an initial estimate yet, so buffer measurements until we
            #can successfully triangulate
            if not other_id in self.initial_ranges.keys():
                self.initial_ranges[other_id] = []
            for si, r in enumerate(message.measurements):
                self.initial_ranges[other_id].append((ind, si, r))
        else:
            self.update_ids.add(other_id)
            other_ind = self.n_poses[other_id]
            for si, r in enumerate(message.measurements):
                #measurements stored as (self_pose_index, other_id,
                #                        other_pose_index, sensor_index, range)
                meas = (ind, other_id, other_ind, si, r)
                self.range_measurements.append(meas)


    def receive_odometry_message(self, message):
        """Receive an odometry message to be able to sense the motion.

        This odometry measurement should be a result of the control output.
        Odometry Format:  [th_dot, x_dot, y_dot]

        Args:
            message: Odometry measurement as a result of control output.
        """
        logger = logging.getLogger(self.logger_name)
        logger.debug("Received odometry message %s", message)
        self.odom_measurements.append(np.expand_dims(message, axis=1))


    def get_short_range_message(self):
        """Get short range message to transmit to another robot.

        This message is allowed to be large, but should only be transmitted when
        the distance between robots is under a certain amount.

        Returns:
            Object to be transmitted to other robots when in short range.
        """

        message = {"id": self.config['id']}
        n = min(self.short_range_history, len(self.odom_measurements))
        message["data"] = self.odom_measurements[-n:]
        self.logger.debug("Returning short range message %s", message)
        return message


    def get_current_goal(self):
        """Get the current goal from the input list of goals.

        The robot configuration includes a list of goals. This function will
        return the proper current goal.

        Returns:
            The current goal of the robot.
        """
        return self.my_goals[self.goal_index]


    def update_goal(self):
        """Update the current goal, if necessary.

        Uses the current position estimate to determine if we're close enough to
        the current goal. If we've stuck around the current goal long enough,
        update to the next goal.
        """

        logger = logging.getLogger(self.logger_name)
        reached = self.get_current_goal() - self.pos
        if np.linalg.norm(reached) <= self.goal_thresh:
            self.at_goal_time += 1
        else:
            self.at_goal_time = 0

        if self.at_goal_time >= self.loiter_time:
            logger.info("Switching to the next goal!")
            self.goal_index += 1
            self.goal_index = min(self.goal_index, len(self.my_goals) - 1)


    def get_long_range_message(self):
        """Get long range message to transmit to another robot.

        This message is to be transmitted along with measurements for long
        range communication. The objects returned from here will be relatively
        small.

        Returns:
            Object to be transmitted to other robots in long range.
        """

        message = {
            "id": self.config['id'],
            "goal": self.get_current_goal(),
        }
        logger = logging.getLogger(self.logger_name)
        logger.debug("Returning long range message %s", message)
        return message


    def pid_controller(self, current, setpoint):
        """Get control output given current state and setpoint.

        This is pulled out of the get_control_output function so that it can be
        used to propagate other robot poses forward as well. This function
        currently only implements a P controller. Control output is currently
        given as [0, dx, dy], which means that orientation is never updated.

        Args:
            current: NumPy array of current [x, y] position
            setpoint: Goal state in [x, y]

        Returns:
            Control output from PID controller.
        """

        error = setpoint - current # Compute error relative to goal
        scale = self.v_lin_max / np.abs(error).max() # Determine scale factor P
        v_lin = error if scale > 1 else error * scale # Apply P controller
        control_output = np.hstack(([0], v_lin)) # Format nicely
        return control_output


    def get_control_output(self):
        """Get robot's control output for the current state.

        The robot may attempt to effect some sort of motion at every time step,
        which will be a result of the control output from this time step.

        Control Input:  [linear_velocity, angular_velocity]

        Returns:
            Control output to feed into the robot model.
        """

        logger = logging.getLogger(self.logger_name)
        self.control_output = self.pid_controller(self.pos,
                self.get_current_goal())
        logger.debug("Returning control output %s", self.control_output)
        return self.control_output


    def iterative_update(self, A, b):
        A_sp = csc_matrix(A.T.dot(A))
        A_splu = splu(A_sp)
        prev_x = self.x
        dx = A_splu.solve(A.T.dot(b))
        if self.meas_err(prev_x + dx) > self.meas_err(prev_x):
          #error went up, stop iterating now
          self.logger.debug("Error went up from %f to %f", self.meas_err(prev_x), self.meas_err(prev_x + dx))
          return False
        self.x = prev_x + dx

        #  logger = logging.getLogger(self.logger_name)
        #  logger.error("WATCH ME: %f, %f", self.x.T.max(), prev_x.T.max())
        if(euclidean(self.x,prev_x) < self.stopping_threshold):
            return False
        return True


    def prior_err(self, x):
        j0 = self.first_pose_ind(self.id)
        p0 = x[j0:j0 + self.pose_dim, 0]
        dz = self.start_pos - p0
        sigma = self.sigma_init_inv.dot(self.sigma_init_inv)
        return dz.T.dot(sigma.dot(dz))

    def odom_err(self, x):
        j_start = self.first_pose_ind(self.id)
        err = 0
        for ind, m in enumerate(self.odom_measurements):
            j0 = j_start + self.pose_dim * ind
            j1 = j_start + self.pose_dim * (ind + 1)
            p0 = x[j0:j0 + self.pose_dim]
            p1 = x[j1:j1 + self.pose_dim]
            #TODO: odom measurements should be multiplied by dt
            mPred = p1 - p0
            mPred[0] = self.wrap_to_pi(mPred[0])
            dz = m - mPred
            dz[0] = self.wrap_to_pi(dz[0])
            sigma = self.sigma_odom_inv.dot(self.sigma_odom_inv)
            err += dz.T.dot(sigma.dot(dz))
        return err

    def other_odom_err(self, x):
        err = 0
        for id in self.n_poses:
            if id == self.id:
                continue
            j_start = self.first_pose_ind(id)
            for ind in range(self.n_poses[id] - 1):
                j0 = j_start + self.pose_dim * ind
                j1 = j_start + self.pose_dim * (ind + 1)
                p0 = x[j0:j0 + self.pose_dim]
                p1 = x[j1:j1 + self.pose_dim]
                mPred = p1 - p0
                mPred[0] = self.wrap_to_pi(mPred[0])
                control = self.other_control[id][ind].reshape((-1, 1))
                dz = control-mPred
                dz[0] = self.wrap_to_pi(dz[0])
                sigma = self.sigma_other_odom_inv.dot(self.sigma_other_odom_inv)
                err += dz.T.dot(sigma.dot(dz))
        return err

    def range_err(self, x):
        err = 0
        for ind, other_id, other_ind, si, rMeas in self.range_measurements:
            j0 = self.first_pose_ind(self.id) + self.pose_dim * ind
            j1 = self.first_pose_ind(other_id) + self.pose_dim * other_ind

            th = x[j0, 0]
            pos0 = self.sensor_pose(ind, si, x)
            pos1 = x[j1+1:j1 + self.pose_dim, 0]
            delta = self.sensor_deltas[si]
            d = pos0 - pos1
            r = np.linalg.norm(d)

            dz = np.array((rMeas - r))
            sigma = self.sigma_range_inv.dot(self.sigma_range_inv)
            err += dz.T.dot(sigma.dot(dz))
        return err

    def meas_err(self, x):
        err = self.prior_err(x)
        err += self.odom_err(x)
        err += self.other_odom_err(x)
        err += self.range_err(x)
        return err

    def build_odom_system(self, A, b, i0):
        """builds the block of A and b corresponding to odometry measurements
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Index of next available row in A
        """
        i = i0
        j_start = self.first_pose_ind(self.id)
        for ind, m in enumerate(self.odom_measurements):
            j0 = j_start + self.pose_dim * ind
            j1 = j_start + self.pose_dim * (ind + 1)
            p0 = self.x[j0:j0 + self.pose_dim]
            p1 = self.x[j1:j1 + self.pose_dim]
            H = np.array([[-1, 0, 0, 1, 0, 0],
                          [0, -1, 0, 0, 1, 0],
                          [0, 0, -1, 0, 0, 1]])
            H = self.sigma_odom_inv.dot(H)
            A[i:i + self.odom_dim, j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
            A[i:i + self.odom_dim, j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
            #TODO: odom measurements should be multiplied by dt
            mPred = p1 - p0
            mPred[0] = self.wrap_to_pi(mPred[0])
            dz = m - mPred
            dz[0] = self.wrap_to_pi(dz[0])
            b[i:i + self.odom_dim, 0] = dz.flatten()
            i += self.odom_dim
        return i


    def build_range_system(self, A, b, i0):
        """Builds the block of A and b corresponding to range measurements
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Index of next available row in A

        """
        i = i0
        for ind, other_id, other_ind, si, rMeas in self.range_measurements:
            j0 = self.first_pose_ind(self.id) + self.pose_dim * ind
            j1 = self.first_pose_ind(other_id) + self.pose_dim * other_ind

            th = self.x[j0, 0]
            pos0 = self.sensor_pose(ind, si)
            pos1 = self.x[j1+1:j1 + self.pose_dim, 0]
            delta = self.sensor_deltas[si]
            d = pos0 - pos1
            r = np.linalg.norm(d)
            H = np.array([[(d[0] * (-delta[0]*np.sin(th) - delta[1]*np.cos(th)) +
                            d[1] * (delta[0]*np.cos(th) - delta[1]*np.sin(th))) / r,
                           d[0] / r, d[1] / r,  0, -d[0] / r, -d[1] / r]])
            H = self.sigma_range_inv.dot(H)

            A[i:i + self.range_dim, j0:j0 + self.pose_dim] = H[0, :self.pose_dim]
            A[i:i + self.range_dim, j1:j1 + self.pose_dim] = H[0, self.pose_dim:]
            b[i:i + self.range_dim, 0] = rMeas - r
            i += self.range_dim
        return i


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
            j_start = self.first_pose_ind(id)
            for ind in range(self.n_poses[id] - 1):
                j0 = j_start + self.pose_dim * ind
                j1 = j_start + self.pose_dim * (ind + 1)
                p0 = self.x[j0:j0 + self.pose_dim]
                p1 = self.x[j1:j1 + self.pose_dim]
                H = np.array([[-1, 0, 0, 1, 0, 0],
                              [0, -1, 0, 0, 1, 0],
                              [0, 0, -1, 0, 0, 1]])
                H = self.sigma_other_odom_inv.dot(H)
                A[i:i + self.odom_dim, j0:j0 + self.pose_dim] = H[:, :self.pose_dim]
                A[i:i + self.odom_dim, j1:j1 + self.pose_dim] = H[:, self.pose_dim:]
                mPred = p1 - p0
                mPred[0] = self.wrap_to_pi(mPred[0])
                control = self.other_control[id][ind].reshape((-1, 1))
                dz = control-mPred
                dz[0] = self.wrap_to_pi(dz[0])
                b[i:i + self.odom_dim, 0] = dz.flatten()
                i += self.odom_dim
        return i


    def build_priors(self, A, b, i0):
        """builds the block of A and b corresponding to priors
        Args:
            A: Linear system being built
            b: Error vector
            i0: row at which to insert this block

        Returns:
            Number of rows in this block
        """

        i = i0
        j0 = self.first_pose_ind(self.id)
        p0 = self.x[j0:j0 + self.pose_dim, 0]
        A[i0:i0 + self.pose_dim, j0:j0 + self.pose_dim] = self.sigma_init_inv
        b[i0:i0 + self.pose_dim, 0] = self.start_pos - p0
        i += self.pose_dim

        if self.fake_thetas:
          #If measurements are insufficient to determine other robot
          #orientations, add a prior setting other robots' initial orientations
          #to 0
          for id in self.n_poses:
              if id == self.id:
                  continue
              j0 = self.first_pose_ind(id)
              A[i, j0] = 1
              b[i, 0] = 0 - self.x[j0]

              i += 1

        return i


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

        i = 0
        i = self.build_priors(A, b, i)
        i = self.build_odom_system(A, b, i)
        i = self.build_other_odom_system(A, b, i)
        i = self.build_range_system(A, b, i)

        return A, b


    def triangulate(self, measurements):
        """Triangulate position of other robot based on sensor readings

        This method is used to estimate an initial position for another robot
        the first time it is seen
        """
        if len(measurements) < 3:
          return None
        j_start = self.first_pose_ind(self.id)
        A = np.zeros((len(measurements) - 1, 2))
        b = np.zeros((len(measurements) - 1, 1))

        #Find x_n, y_n, r_n
        ind_n, si_n, r_n = measurements[-1]
        x_n, y_n = self.sensor_pose(ind_n, si_n)

        for i, (ind, si, r) in enumerate(measurements[:-1]):
            x, y = self.sensor_pose(ind, si)
            A[i, :] = (2*(x_n - x), 2*(y_n - y))
            b[i] = r**2 - r_n**2 - x**2 + x_n**2 - y**2 + y_n**2

        pos, res, R, s = np.linalg.lstsq(A, b)
        if R < 2:
          return None
        else:
          return pos


    def compute(self):
        """Perform all the computation required to process messages.

        This method is called every time step, before time is updated in the
        robot (before .step()). This method should be the one the robot uses to
        perform all of the SLAM updates.
        """
        logger = logging.getLogger(self.logger_name)
        logger.debug("Computing at time %d", self.t)

        #use previous pose as current pose estimate
        j0 = self.last_pose_ind(self.id)
        p_new = self.x[j0:j0+self.pose_dim] + self.odom_measurements[-1]
        self.x = np.insert(self.x, j0+self.pose_dim, p_new, axis=0)
        self.n_poses[self.id] += 1

        #Try to estimate the initial position of newly visible robots
        done_triangulating = []

        # For all other robots (which have already been localized), update their
        # poses, at exactly the same place (for now).
        for other_id in self.update_ids:
            j0 = self.last_pose_ind(other_id)
            previous_pose = self.x[j0:j0+self.pose_dim]

            # To compute the update, take the previous position and current goal
            # information and pass them through the PID controller. Treat that
            # output as the path that we think the robot must have taken.
            previous_pos = previous_pose[1:].reshape(-1)
            control = self.pid_controller(previous_pos, self.goals[other_id])
            current_pose = previous_pose + control.reshape(-1, 1) # PID update

            if not other_id in self.other_control:
                self.other_control[other_id] = []
            self.other_control[other_id].append(control)

            self.x = np.insert(self.x, j0 + self.pose_dim, current_pose, axis=0)
            self.n_poses[other_id] += 1

        # First time you see each robot (with enough measurements), do some
        # triangulation.
        for other_id in self.initial_ranges.keys():
            pos = self.triangulate(self.initial_ranges[other_id])
            if pos is not None: #Triangulation was successful
                done_triangulating.append(other_id)
                #Use pos as initial state estimate (set th = 0)
                j0 = self.first_pose_ind(other_id)
                p_new = np.vstack(([0], pos))
                self.x = np.insert(self.x, j0, p_new, axis=0)
                self.n_poses[other_id] = 1
                #self.update_ids.add(other_id)

                #Use the most recent measurement for SLAM
                #TODO: Should probably incorporate all measurements for SLAM
                ind = self.n_poses[self.id] - 1
                measurements = [(i, r) for _ind, i, r in
                                self.initial_ranges[other_id] if _ind == ind]
                for i, r in measurements:
                    #measurements stored as (self_pose_index, other_id,
                    #                        other_pose_index, sensor_index,
                    #                        range)
                    meas = (ind, other_id, 0, i, r)
                    self.range_measurements.append(meas)

        # Coupled with the above, delete robot IDs which have been triangulated.
        for other_id in done_triangulating:
            del self.initial_ranges[other_id]


        # Build the system to solve.
        for i in range(0,self.max_iterations):
            A,b = self.build_system()
            if(not self.iterative_update(A,b)):
                break

        # Reset update_ids.
        self.update_ids = set()

        # Update the goals, if necessary.
        self.update_goal()


    def step(self, step):
        """Increment the robot's internal time state by some step size.

        Args:
            step: time to be added to internal state
        """
        self.t += step
        logger = logging.getLogger(self.logger_name)
        logger.debug("Adding time %d to get new time %d", step, self.t)
