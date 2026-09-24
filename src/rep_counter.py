class RepCounter:
    """
    Tracks squat repetitions based on knee angle over time.

    States:
    - "up": standing, knee angle is high (~160+ degrees)
    - "down": squatting, knee angle is low (~100 or less)

    A rep is counted each time we go from "down" back to "up".
    """

    def __init__(self, up_threshold=160, down_threshold=100):
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.stage = "up"   # current position state
        self.count = 0

    def update(self, knee_angle):
        """
        Call this every frame with the current knee angle.
        Returns the updated rep count.
        """
        if knee_angle > self.up_threshold:
            if self.stage == "down":
                # completed a full down -> up cycle
                self.count += 1
            self.stage = "up"

        elif knee_angle < self.down_threshold:
            self.stage = "down"

        return self.count

    def get_stage(self):
        return self.stage

    def get_count(self):
        return self.count