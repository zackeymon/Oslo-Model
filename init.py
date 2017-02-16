import numpy as np
import matplotlib.pyplot as plt
import pickle
import analysis
from collections import OrderedDict
from pile import Pile

# CONSTANTS
POSSIBLE_THRESHOLD_SLOPES = (1, 2)
OSLO_PROBS = (0.5, 0.5)

data_dict = pickle.load(open('height_and_avalanche_data', 'rb'))
