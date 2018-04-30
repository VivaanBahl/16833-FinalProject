import atexit
import logging
import pdb
import time

import disturbances as db
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.animation as manimation
import numpy as np

class Visualizer(object):
    def __init__(self, num_robots, disturbance):
        """Initialize visualizer with program data.

        Args:
            num_robots - number of robots in the system
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting visualization.")
        
        # Note that these are aliased to the real ones. Don't modify, just read.
        self.main_figure_number = 1
        self.error_figure_number = 2

        # Start the stuff for the movie writer.
        metadata = dict(title='MAS SLAM Simulation', artist='Awesome Squad')
        self.writer = manimation.writers['ffmpeg'](fps=10, metadata=metadata)
        name = "MAS" + "_" + str(int(round(time.time()))) + ".mp4"
        self.writer.setup(plt.figure(self.main_figure_number), name, dpi=200)
        atexit.register(self.cleanup) # Close the writer when python terminates

        # Initialize errors
        self.robot_errors = []
        for i in range(num_robots):
            self.robot_errors.append([])

        if disturbance == 'radial_waves':
          self.disturbance = db.radial_waves
        elif disturbance == 'linear':
          self.disturbance = db.linear
        else:
          self.disturbance = db.no_force

        self.view_mode = "past_trajectories" # or None
        # self.view_mode = None

        # Create the figures
        fig = plt.figure(self.main_figure_number)
        plt.ion()
        plt.show()
        
        # fig2 = plt.figure(self.error_figure_number)
        # plt.ion()
        
        self.x_min_coord = 0
        self.x_max_coord = 0
        self.y_min_coord = 0
        self.y_max_coord = 0

    def cleanup(self):
        """Clean up visualization objects.

        Specifically, close the movie writer.
        """
        self.logger.info("Finishing writing movie!")
        self.writer.finish()


    def draw_messages(self, ax, motions, measurement_arr, color):
        for (xmit_index, recv_index) in measurement_arr:
            motion1 = motions[xmit_index]
            motion2 = motions[recv_index]
            pos1 = motion1.pos
            pos2 = motion2.pos

            arrow_origin = pos1
            arrow_size = pos2 - pos1

            ax.arrow(arrow_origin[0], arrow_origin[1], arrow_size[0], arrow_size[1], ec=color)

    def get_euclidean_error(self, robot_gt, robot_belief):
        """
        Calculates the Euclidean distance betweeen a robot's ground truth and it's current pose
        Both robot_gt and robot_belief are passed in as column numpy arrays
        """
        return np.sqrt(np.sum(np.square(robot_gt - robot_belief)))


    def update(self, robots, motions, short_range_measurements, long_range_measurements):
        """Perform visualization updates here."""

        self.logger.debug("Updating visualization.")

        # TODO convert to numpy arrays for scalability
        x_robot_gt = []
        y_robot_gt = []
        x_goal = []
        y_goal = []
        headings = []

        if(self.view_mode == "past_trajectories"):
            max_num_poses = 0
            for i in range(0,len(robots)):
                for j in range(0,len(robots)):
                    max_num_poses = max(max_num_poses, robots[i].n_poses[j])


            # create square matrix showing all the robots' beliefs about each other
            robot_beliefs = np.zeros([len(robots), len(robots), max_num_poses, 2])  # 2 bc of x and y

            for i, motion in enumerate(motions):
                self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

                pos = motion.pos
                x_robot_gt.append(pos[0])
                y_robot_gt.append(pos[1])

                robot = robots[i]

                for goal_index in range(0, robot.goal_index + 1):
                    goal = robot.my_goals[goal_index]
                    x_goal.append(goal[0])
                    y_goal.append(goal[1])

                for other_robot_id in range(0, len(robots)):
                    if other_robot_id in robot.n_poses:
                        for pose_index in range(0, robot.n_poses[other_robot_id]):

                            start = sum([robot.n_poses[id] for id in robot.n_poses if id < other_robot_id])

                            j0 = robot.pose_dim * (start + pose_index)

                            robot_beliefs[i, other_robot_id, pose_index, 0] = robot.x[j0 + 1]
                            robot_beliefs[i, other_robot_id, pose_index, 1] = robot.x[j0 + 2]

                self.robot_errors[i].append(self.get_euclidean_error(pos, robot_beliefs[i, i, robot.n_poses[i] -1, :]))

                vel = motion.vel
                headings.append((vel[0], vel[1]))

            # set the viewport bounds according to all the beliefs and goals
            x_min_coord_temp = min(np.min(robot_beliefs[:,:,:,0]), min(x_goal))
            x_max_coord_temp = max(np.max(robot_beliefs[:,:,:,0]), max(x_goal))
            y_min_coord_temp = min(np.min(robot_beliefs[:,:,:,1]), min(y_goal))
            y_max_coord_temp = max(np.max(robot_beliefs[:,:,:,1]), max(y_goal))

        else:

            robot_beliefs = np.zeros([len(robots), len(robots), 2])  # 2 bc of x and y

            for i, motion in enumerate(motions):
                self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

                pos = motion.pos
                x_robot_gt.append(pos[0])
                y_robot_gt.append(pos[1])

                robot = robots[i]

                for goal_index in range(0, robot.goal_index + 1):
                    goal = robot.my_goals[goal_index]
                    x_goal.append(goal[0])
                    y_goal.append(goal[1])

                for other_robot_id in range(0, len(robots)):
                    (other_bel_x, other_bel_y) = robot.robot_pos(other_robot_id)
                    if other_bel_x is not None:
                        robot_beliefs[i, other_robot_id, 0] = other_bel_x
                        robot_beliefs[i, other_robot_id, 1] = other_bel_y

                self.robot_errors[i].append(self.get_euclidean_error(pos, robot_beliefs[i, i, :]))

                vel = motion.vel
                headings.append((vel[0], vel[1]))

            # set the viewport bounds according to all the beliefs and goals
            x_min_coord_temp = min(np.min(robot_beliefs[:,:,0]), min(x_goal))
            x_max_coord_temp = max(np.max(robot_beliefs[:,:,0]), max(x_goal))
            y_min_coord_temp = min(np.min(robot_beliefs[:,:,1]), min(y_goal))
            y_max_coord_temp = max(np.max(robot_beliefs[:,:,1]), max(y_goal))

        # get the main update figure, clear it out for our update
        # fig = plt.figure(self.main_figure_number)
        # fig.clf()

        # draw the error graphs
        # error_fig = plt.figure(self.error_figure_number)
        # error_fig.clf()
        # num_subplots = len(robots)
        # for i in range(1, num_subplots + 1):
        #     plt.subplot(num_subplots, 1, i)
        #     plt.plot(self.robot_errors[i-1])
        
        # get the main update figure, clear it out for our update
        fig = plt.figure(self.main_figure_number)
        fig.clf()

        # calculate viewport bounds

        t1 = 2
        t2 = 2

        self.x_min_coord = min(self.x_min_coord, (np.floor(x_min_coord_temp/t1)*t1)-t2)
        self.x_max_coord = max(self.x_max_coord, (np.ceil( x_max_coord_temp/t1)*t1)+t2)
        self.y_min_coord = min(self.y_min_coord, (np.floor(y_min_coord_temp/t1)*t1)-t2)
        self.y_max_coord = max(self.y_max_coord, (np.ceil( y_max_coord_temp/t1)*t1)+t2)

        x_scale = self.x_max_coord - self.x_min_coord
        y_scale = self.y_max_coord - self.y_min_coord

        #make the scale the same on each axis
        if(x_scale > y_scale):
            self.y_max_coord = ((self.y_max_coord + self.y_min_coord)/2) + (x_scale/2)
            self.y_min_coord = self.y_max_coord - x_scale
        else:
            self.x_max_coord = ((self.x_max_coord + self.x_min_coord)/2) + (y_scale/2)
            self.x_min_coord = self.x_max_coord - y_scale

        plt.xlim([self.x_min_coord, self.x_max_coord])
        plt.ylim([self.y_min_coord, self.y_max_coord])

        # plt.xlim([self.x_min_coord - margin, self.x_max_coord + margin])
        # plt.ylim([self.y_min_coord - margin, self.y_max_coord + margin])

        #draw force fields

        X, Y = np.meshgrid(np.arange(self.x_min_coord, self.x_max_coord,1), 
                           np.arange(self.y_min_coord, self.y_max_coord,1))
        U = db.linear(0*X,X,Y)[1]
        V = db.linear(0*X,X,Y)[2]
        Q = plt.quiver(X, Y, U, V, units='width', color=(0.0, 0.0, 0.0, 0.3))

        # draw robots ground truth
        plt.scatter(x_robot_gt, y_robot_gt, c='k', marker='o')

        if(self.view_mode == "past_trajectories"):
            # draw the robot belief as their own individual colors
            for i in range(0, len(robots)):
                belief_x = np.ndarray([])
                belief_y = np.ndarray([])

                i_x = 0
                i_y = 0

                for other_robot_id in range(0, len(robots)):
                    belief_x = np.append(belief_x, robot_beliefs[i, other_robot_id, 0:robots[i].n_poses[other_robot_id], 0])
                    belief_y = np.append(belief_y, robot_beliefs[i, other_robot_id, 0:robots[i].n_poses[other_robot_id], 1])

                    if(other_robot_id == i):
                        i_x = len(belief_x)-1
                        i_y = len(belief_y)-1

                
                plt.scatter(belief_x, belief_y, marker='o')

                plt.annotate(xy=(belief_x[i_x], belief_y[i_y]), s="{}".format(i))

        else:
            for i in range(0, len(robots)):
                robot_i_beliefs_x = robot_beliefs[i, :, 0]
                robot_i_beliefs_y = robot_beliefs[i, :, 1]
                plt.scatter(robot_i_beliefs_x, robot_i_beliefs_y, marker='o')
                for j in range(len(robot_i_beliefs_x)):
                    plt.annotate(xy=(robot_i_beliefs_x[j], robot_i_beliefs_y[j]), s="{}".format(j))

        #draw legends
        robot_legend_labels = ["Robot {} beliefs".format(i) for i in range(0, len(robots))]
        robot_legend_labels = ["Disturbance Field", "Ground Truths"] + robot_legend_labels
        plt.legend(robot_legend_labels,loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox = True, ncol = 2)

        # draw "headings" aka the velocity vectors
        ax = plt.axes()
        for i, heading in enumerate(headings):
           ax.arrow(x_robot_gt[i], y_robot_gt[i], heading[0], heading[1])

        # draw goals as green x's
        plt.scatter(x_goal, y_goal, c='g', marker='x')

        # draw messages being sent between robots
        self.draw_messages(ax, motions, short_range_measurements, 'r')
        self.draw_messages(ax, motions, long_range_measurements, 'y')



        plt.pause(0.01)

        self.logger.debug("Writing frame to file!")
        self.writer.grab_frame()
        
        plt.show()
