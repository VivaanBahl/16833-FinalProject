import logging
import numpy as np

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
        self.pos = config['start']

        self.t = 0


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


    def receive_odometry_message(self, message):
        """Receive an odometry message to be able to sense the motion.

        This odometry measurement should be a result of the control output.

        Args:
            message: Odometry measurement as a result of control output.
        """
        self.logger.debug("Received odometry message %s", message)
        self.pos += message


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

        Returns:
            Control output to feed into the robot model.
        """

        dir = self.goal - self.pos
        control_output = dir / np.linalg.norm(dir)
        self.logger.debug("Returning control output %s", control_output)
        return control_output


    def compute(self):
        """Perform all the computation required to process messages.

        This method is called every time step, before time is updated in the
        robot (before .step()). This method should be the one the robot uses to
        perform all of the SLAM updates.
        """
        self.logger.debug("Computing at time %d", self.t)
        pass


    def step(self, step):
        """Increment the robot's internal time state by some step size.

        Args:
            step: time to be added to internal state
        """
        self.t += step
        self.logger.debug("Adding time %d to get new time %d", step, self.t)
