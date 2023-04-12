import numpy as np
import pdb


class Explitcitness():
    def __init__(self, mode='baseline'):
        self.curves = []
        self.curves_names = []
        self.baseline_losses = []

        self.mode = mode
    def add_curve(self, x,y, baseline_loss=1.0, name='curve'):
        x = np.array(x)
        y = np.array(y)

        # we use the minimum loss achieved until that capacity
        for i in range(y.shape[0]):
            y[i] = y[:(i+1)].min()

        curve = np.vstack((x,y))
        self.curves.append(curve)
        self.baseline_losses.append(baseline_loss)
        self.curves_names.append(name)
    
    def get_explitcitness(self, debug=False):
        if len(self.curves) == 0:
            return {}
        max_x = np.array([c[0].max() for c in self.curves]).max()
        min_x = np.array([c[0].min() for c in self.curves]).min()

        max_y = np.array([c[1].max() for c in self.curves]).max()
        min_y = np.array([c[1].min() for c in self.curves]).min()

        for ind_c, curve in enumerate(self.curves):
            new_x = list(curve[0])
            new_y = list(curve[1])
            # we add a virtual point with maximum capacity but same performance as the previous one
            # we assume we have reached convergence
            if curve[0,-1] < max_x:
                new_x.append(max_x)
                new_y.append(new_y[-1])
        
            new_x = np.array(new_x)
            new_y = np.array(new_y)
            new_curve = np.vstack((new_x,new_y))
            self.curves[ind_c] = np.array(new_curve)

        all_E = {}
        for i, name in enumerate(self.curves_names):
            # this is used in the iclr paper. we assume l* = 0
            max_area = (max_x - min_x) * self.baseline_losses[i] 
            E = compute_explitcitness(self.curves[i][0], self.curves[i][1], 
                global_max_area = max_area,
                baseline_loss=self.baseline_losses[i],
                name=name,
                debug=debug
                )

            all_E[name] = E
        return all_E


def compute_explitcitness(x,y, max_area=None, max_x = None, min_x=None, baseline_loss = 1.0, name='',
        debug=False):
    x = np.array(x)
    y = np.array(y)

    min_y_index = np.argmin(y)
    min_y = y[min_y_index]

    dy = 0.5 * (y[:-1] - min_y) + 0.5 * (y[1:] - min_y)  
    dx = x[1:] - x[:-1]

    area_under = dy * dx
    E = 2 * area_under / max_area
    E = E.sum()
    E = 1.0 - E

    return E
