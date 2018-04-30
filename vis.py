import matplotlib.pyplot as plt
import numpy as np
import logging
import disturbances as db
import pdb


class Visualizer(object):
    def __init__(self, num_robots):
        """Initialize visualizer with program data.

        Args:
            num_robots - number of robots in the system
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting visualization.")

        # Note that these are aliased to the real ones. Don't modify, just read.
        self.main_figure_number = 1
        self.error_figure_number = 2

        self.robot_errors = []
        for i in range(num_robots):
            self.robot_errors.append([])

        fig = plt.figure(self.main_figure_number)
        plt.ion()
        plt.show()
        fig2 = plt.figure(self.error_figure_number)
        plt.ion()


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

        # get the main update figure, clear it out for our update
        fig = plt.figure(self.main_figure_number)
        fig.clf()

        # TODO convert to numpy arrays for scalability
        x_robot_gt = []
        y_robot_gt = []
        x_goal = []
        y_goal = []
        headings = []

        # create square matrix showing all the robots' beliefs about each other
        robot_beliefs = np.zeros([len(robots), len(robots), 2])  # 2 bc of x and y

        # default viewport for the visualizer
        x_min_coord = 0
        x_max_coord = 0
        y_min_coord = 0
        y_max_coord = 0

        for i, motion in enumerate(motions):
            self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

            pos = motion.pos
            x_robot_gt.append(pos[0])
            y_robot_gt.append(pos[1])

            robot = robots[i]
            goal = robot.goal
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

        x_min_coord = x_min_coord_temp if x_min_coord_temp < x_min_coord else x_min_coord 
        x_max_coord = x_max_coord_temp if x_max_coord_temp > x_max_coord else x_max_coord
        y_min_coord = y_min_coord_temp if y_min_coord_temp < y_min_coord else y_min_coord
        y_max_coord = y_max_coord_temp if y_max_coord_temp > y_max_coord else y_max_coord

        #draw force fields
        X, Y = np.meshgrid(np.arange(x_min_coord, x_max_coord, (x_max_coord - x_min_coord)/50), np.arange(y_min_coord, y_max_coord, (y_max_coord - y_min_coord)/50))
        U = db.linear(0*X,X,Y)[1]
        V = db.linear(0*X,X,Y)[2]
        Q = plt.quiver(X, Y, U, V, units='width', color=(0.0, 0.0, 0.0, 0.3))


        # draw robots ground truth
        plt.scatter(x_robot_gt, y_robot_gt, c='k', marker='o')

        # draw the robot belief as their own individual colors
        for i in range(0, len(robots)):
            robot_i_beliefs_x = robot_beliefs[i, :, 0]
            robot_i_beliefs_y = robot_beliefs[i, :, 1]
            plt.scatter(robot_i_beliefs_x, robot_i_beliefs_y, marker='o')

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

        # calculate viewport bounds
        x_scale = x_max_coord - x_min_coord
        y_scale = y_max_coord - y_min_coord
        x_margin = x_scale / 12
        y_margin = y_scale / 12
        plt.xlim([x_min_coord - x_margin, x_max_coord + x_margin])
        plt.ylim([y_min_coord - y_margin, y_max_coord + y_margin])

        # draw the error graphs
        error_fig = plt.figure(self.error_figure_number)
        error_fig.clf()
        num_subplots = len(robots)
        for i in range(1, num_subplots + 1):
            plt.subplot(num_subplots, 1, i)
            plt.plot(self.robot_errors[i-1])

        plt.pause(0.05)
        plt.show()
