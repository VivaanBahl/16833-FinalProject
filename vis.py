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

        # default viewport for the visualizer
        x_min_coord = 0
        x_max_coord = 0
        y_min_coord = 0
        y_max_coord = 0

        for i, motion in enumerate(self.motions):
            self.logger.debug("Robot {} has pos {}, vel {}".format(i, motion.pos, motion.vel))

            pos = motion.pos
            x_robot.append(pos[0])
            y_robot.append(pos[1])

            robot = self.robots[i]
            goal = robot.goal
            x_goal.append(goal[0])
            y_goal.append(goal[1])

            # if a target is outside of the viewport, set viewport to include it
            if pos[0] < x_min_coord or goal[0] < x_min_coord:
                x_min_coord = min(pos[0], goal[0])
            if pos[0] > x_max_coord or goal[0] > x_max_coord:
                x_max_coord = max(pos[0], goal[0])
            if pos[1] < y_min_coord or goal[1] < y_min_coord:
                y_min_coord = min(pos[1], goal[1])
            if pos[1] > y_max_coord or goal[1] > y_max_coord:
                y_max_coord = max(pos[1], goal[1])

        # draw robots
        plt.scatter(x_robot, y_robot)

        # draw "headings" aka the velocity vectors
        ax = plt.axes()

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
