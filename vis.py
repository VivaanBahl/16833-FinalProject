import matplotlib.pyplot as plt
import logging


class Visualizer(object):
    def __init__(self):
        """Initialize visualizer with program data.

        Args:
            robots: List of Robot objects.
            motions: List of RobotMotion objects.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting visualization.")

        # Note that these are aliased to the real ones. Don't modify, just read.
        self.figure_number = 1

        fig = plt.figure(self.figure_number)
        plt.ion()
        plt.show()

    def update(self, robots, motions, odometry):
        """Perform visualization updates here."""

        self.logger.debug("Updating visualization.")

        # get the main update figure, clear it out for our update
        fig = plt.figure(self.figure_number)
        fig.clf()

        # TODO convert to numpy arrays for scalability
        x_robot_gt = []
        y_robot_gt = []
        x_robot_pre = []
        y_robot_pre = []
        x_goal = []
        y_goal = []
        headings = []

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

            vel = motion.vel
            headings.append((vel[0], vel[1]))

            odom = odometry[i]
            x_robot_pre.append(pos[0] + odom[0])
            y_robot_pre.append(pos[1] + odom[1])

            # if a target is outside of the viewport, set viewport to include it
            if pos[0] < x_min_coord or goal[0] < x_min_coord:
                x_min_coord = min(pos[0], goal[0])
            if pos[0] > x_max_coord or goal[0] > x_max_coord:
                x_max_coord = max(pos[0], goal[0])
            if pos[1] < y_min_coord or goal[1] < y_min_coord:
                y_min_coord = min(pos[1], goal[1])
            if pos[1] > y_max_coord or goal[1] > y_max_coord:
                y_max_coord = max(pos[1], goal[1])

        # draw robots ground truth
        plt.scatter(x_robot_gt, y_robot_gt, c='k', marker='o')

        # draw the robot's belief
        plt.scatter(x_robot_pre, y_robot_pre, c='b', marker='o')

        # draw "headings" aka the velocity vectors
        ax = plt.axes()
        for i, heading in enumerate(headings):
           ax.arrow(x_robot_gt[i], y_robot_gt[i], heading[0], heading[1])

        # draw goals as green x's
        plt.scatter(x_goal, y_goal, c='g', marker='x')

        # calculate viewport bounds
        x_scale = x_max_coord - x_min_coord
        y_scale = y_max_coord - y_min_coord
        x_margin = x_scale / 12
        y_margin = y_scale / 12
        plt.xlim([x_min_coord - x_margin, x_max_coord + x_margin])
        plt.ylim([y_min_coord - y_margin, y_max_coord + y_margin])

        plt.pause(0.05)
        plt.show()
