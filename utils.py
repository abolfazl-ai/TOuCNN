import time
from Plotter import Plotter


def iter_print(iteration, x, history, plot=None):
    if iteration % 5 == 0:
        print(f'{iteration:03}     '
              f'{history["Loss"][-1]:<12.5f}{history["Objective"][-1]:<12.5f}'
              f'{history["Volume"][-1]:<10.3f}{history["Mass"][-1]:<10.3f}{history["Cost"][-1]:<10.3f}'
              f'{history["GreyElements"][-1]:<12.5f}'
              f'{history["alpha"][-1]:<12.2f}{history["beta"][-1]:<11.2f}{history["penalty"][-1]:<9.2f}')
    if plot is not None:
        plot.show(x, interactive=True, title=f'Epoch {iteration:03}:   '
                                             f'J={history["Objective"][-1]:0.2f}   '
                                             f'V={history["Volume"][-1]:0.2f}   '
                                             f'M={history["Mass"][-1]:0.2f}   '
                                             f'C={history["Cost"][-1]:0.2f}')


def optimize(optimizer, materials):
    p = Plotter(optimizer.x, materials)
    start = time.time()
    print('Optimization started.')
    print("Epoch   Loss        Objective   Volume    Mass      Cost      "
          "Grey        Alpha       Beta       Penalty")
    ro = optimizer.optimize(lambda i, x, history: iter_print(i, x, history, p))
    print(f'Model converged in {(time.time() - start):0.2f} seconds.')
    p.show_plots(optimizer.history)
    p.show(ro, interactive=False)
