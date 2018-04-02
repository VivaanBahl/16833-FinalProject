import matplotlib.pyplot as plt
import logging


class Visualizer(object):
    def __init__(self, robots, motions):
        """Initialize visualizer with program data.

        Args:
            robots: List of Robot objects.
            motions: List of RobotMotion objects.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Starting visualization.")

        # Note that these are aliased to the real ones. Don't modify, just read.
        self.robots = robots
        self.motions = motions
        self.figure_number = 1

        self.viewport_x_min = -100
        self.viewport_x_max = 100
        self.viewport_y_min = -100
        self.viewport_y_max = 100

        fig = plt.figure(self.figure_number)
        plt.ion()
        plt.show()

    def update(self):
        """Perform visualization updates here."""

        self.logger.debug("Updating visualization.")

        fig = plt.figure(self.figure_number)
        fig.clf()
        x = []
        y = []

        for i, motion in enumerate(self.motions):
            self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

            pos = motion.pos
            x.append(pos[0])
            y.append(pos[1])

        plt.scatter(x, y)
        plt.xlim([self.viewport_x_min, self.viewport_x_max])
        plt.ylim([self.viewport_y_min, self.viewport_y_max])

        x_goal = []
        y_goal = []
        for robot in self.robots:
            goal = robot.goal
            x_goal.append(goal[0])
            y_goal.append(goal[1])

        plt.scatter(x_goal, y_goal, c='g', marker='x')
        plt.pause(0.05)
        plt.show()
