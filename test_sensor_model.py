import numpy as np
import math

from sensor_model import SensorModel
from robot_motion import RobotMotion

config = {
        'sensor_sigma': 0.1,
        'sensor_parameters': [
                {
                    'delta': [1, 0],
                    'orientation': 0,
                    'fov': 180,
                },
                {
                    'delta': [-1, 0],
                    'orientation': 180,
                    'fov': 180,
                },
            ]
}
sm = SensorModel(config)

config1 = {
        'start': [0, 0],
        'sigma_initial': [[0.1, 0],
                          [0, 0.1]],
        'id': 1
}
motion1 = RobotMotion(config1)
motion1.pos = np.array([0, 0])
motion1.th = math.radians(0)

motion2 = RobotMotion(config1)
motion2.pos = np.array([10, 0])

print(sm.get_measurement(motion1, 0, motion2))
print()
print(sm.get_measurement(motion1, 1, motion2))
