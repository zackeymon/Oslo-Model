from init import *
from timeit import timeit as t_it

oslo_pile_xs = Pile(16, OSLO_PROBS, POSSIBLE_THRESHOLD_SLOPES)
oslo_pile_s = Pile(32, OSLO_PROBS, POSSIBLE_THRESHOLD_SLOPES)
oslo_pile_m = Pile(64, OSLO_PROBS, POSSIBLE_THRESHOLD_SLOPES)
oslo_pile_l = Pile(128, OSLO_PROBS, POSSIBLE_THRESHOLD_SLOPES)
oslo_pile_xl = Pile(256, OSLO_PROBS, POSSIBLE_THRESHOLD_SLOPES)

for i in (oslo_pile_xs, oslo_pile_s, oslo_pile_m, oslo_pile_l, oslo_pile_xl):
    pile_heights = []
    for t in range():
        pile_heights[t] = i.get_pile_height()
        i.drop_grain()

    plt.plot(ts, pile_heights, label=i.length)

plt.legend(loc=0)


plt.show()
