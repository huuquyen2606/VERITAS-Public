class LinearSchedule:
    def __init__(self, start_value: float, end_value: float, duration: int):
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration

    def value(self, t: int) -> float:
        if t >= self.duration:
            return self.end_value

        slope = (self.end_value - self.start_value) / self.duration
        return self.start_value + slope * t
