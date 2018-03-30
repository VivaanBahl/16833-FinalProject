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


    def update(self):
        """Perform visualization updates here."""

        self.logger.debug("Updating visualization.")
        
        for motion in self.motions:
            print(motion.pos)
