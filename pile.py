import numpy as np
from my_site import Site


class Pile:
    def __init__(self, length, probs, threshold_zs, name=None):
        choice_args = dict(a=threshold_zs, p=probs)
        self.lattice = np.array([Site(choice_args) for _ in range(length)])
        self.length = length
        self.ava_size = 0
        self.name = name
        self.is_at_steady_state = False

    def reset(self):
        for site in self.lattice:
            site.reset()
        self.is_at_steady_state = False

    def get_pile_height(self):
        """Returns the height of 0th site, which is defined as the pile height"""
        return self.lattice[0].height

    def get_heights(self):
        return [i.height for i in self.lattice]

    def get_threshold_slopes(self):
        return [i.threshold_slope for i in self.lattice]

    def relax(self, site_index):
        self.lattice[site_index].lose_grain()
        self.ava_size += 1

        # toppled off the last site
        if site_index + 1 == self.length:
            self.is_at_steady_state = True
            return

        self.lattice[site_index + 1].add_grain()

    def find_unstable_site_indices(self):
        """Returns a list of indices of the unstable sites"""
        current_slopes = np.append(self.lattice[:-1] - self.lattice[1:], self.lattice[-1].height)
        return [i for i, site in enumerate(self.lattice) if current_slopes[i] > site.threshold_slope]

    def drop_grain(self, site_index=0):
        """Add a grain to a specific site. Continue simulation until pile is stable."""
        # reset avalanche size counter
        self.ava_size = 0

        # drive
        self.lattice[site_index].add_grain()

        while True:
            # find sites that need relaxing
            unstable_site_indices = self.find_unstable_site_indices()
            if not unstable_site_indices:
                # pile stable, stop iterating
                break

            # relax unstable sites
            for i in unstable_site_indices:
                self.relax(i)




