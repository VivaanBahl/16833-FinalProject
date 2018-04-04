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

        fig = plt.figure(self.figure_number)
        plt.ion()
        plt.show()

    def update(self):
        """Perform visualization updates here."""

        self.logger.debug("Updating visualization.")

        # get the main update figure, clear it out for our update
        fig = plt.figure(self.figure_number)
        fig.clf()

        # TODO convert to numpy arrays for scalability
        x_robot = []
        y_robot = []
        x_goal = []
        y_goal = []

        for i, motion in enumerate(self.motions):
            self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

            pos = motion.pos
            x_robot.append(pos[0])
            y_robot.append(pos[1])

            robot = self.robots[i]
            goal = robot.goal
            x_goal.append(goal[0])
            y_goal.append(goal[1])

        # draw robots
        plt.scatter(x_robot, y_robot)

        # draw goals as green x's
        plt.scatter(x_goal, y_goal, c='g', marker='x')

        plt.xlim([self.viewport_x_min, self.viewport_x_max])
        plt.ylim([self.viewport_y_min, self.viewport_y_max])
        plt.pause(0.05)
        plt.show()
