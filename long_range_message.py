class LongRangeMessage(object):
    """Contains message data and distance data"""

    def __init__(self, data, measurements):
        """Store the data given.

        Args:
            data: Message output from get_long_range_message.
            measurement: Measurement from all of this robot's sensors to the
                other robot (which sent the data).
        """
        self.data = data
        self.measurements = measurements


    def __str__(self):
        """Human readable string representation of message"""
        return "Long Range Message: %s, %s" % (self.data, self.measurements)
