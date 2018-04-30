import math
import numpy as np
import pdb

class SensorModel(object):
    def __init__(self, config):
        """Initialize sensor model object.

        Args:
            config: Configuration file.
        """
        self.config = config


    def wrapToPi(self, angle):
        """Wrap a measurement in radians to [-pi, pi]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi


    def get_measurement(self, motion1, i, motion2):
        """Get a distance measurement from sensor i on robot 1 to robot 2.

        Args:
            motion1: RobotMotion object corresponding to robot 1.
            i: Index of sensor to check on robot 1.
            motion2: RobotMotion object corresponding to robot 2.

        Returns:
            None if the robot isn't able to be seen.
            Sampled distance if the robot is able to be seen.
        """
        # First, we need to determine the true position of sensor i in the world
        # frame, which will depend on robot1's orientation.

        # Create the rotation matrix to rotate the delta.
        th = motion1.th
        rot = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th),  np.cos(th)]])

        # Position of the sensor in the world frame.
        sensor = self.config['sensor_parameters'][i]
        sensor_pos = motion1.pos + rot.dot(np.array(sensor['delta'], dtype=float))
        #  print("Sensor %d position: %s" % (i, sensor_pos))

        # Compute the vector to the other robot
        delta = motion2.pos - sensor_pos
        #  print("Delta:", delta)

        # Determine the angle to the other robot
        angle = self.wrapToPi(math.atan2(delta[1], delta[0]))
        #  print("Angle: ", math.degrees(angle))

        fov_center = self.wrapToPi(th + math.radians(sensor['orientation']))
        #  print("Fov center:", math.degrees(fov_center))

        # Check the difference between fov_center and angle.
        angle_diff = self.wrapToPi(fov_center - angle)
        #  print("Angle diff: ", angle_diff)

        if abs(angle_diff) <= math.radians(sensor['fov']) / 2:
            # We can see the robot!
            return np.random.normal(np.linalg.norm(delta),
                                    self.config['sensor_sigma'])
        return None
