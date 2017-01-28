from numpy.random import choice


class Site:
    def __init__(self, choice_parameters, height=0):
        self.height = height
        self.choice_parameters = choice_parameters
        self.threshold_slope = choice(**choice_parameters)

    def __add__(self, other):
        return self.height + other.height

    def __sub__(self, other):
        return self.height - other.height

    def add_grain(self):
        self.height += 1

    def lose_grain(self):
        self.height -= 1
        self.threshold_slope = choice(**self.choice_parameters)
