from Plotter import Plotter


def iter_print(iteration, x, history, plot=None):
    print(f'{iteration:03}     {history["Objective"][-1]:<12.5f}'
          f'{history["Volume"][-1]:<10.3f}{history["Convergence"][-1]:<12.5f}')
    if plot is not None:
        plot.show(x, interactive=True, title=f'Iteration {iteration:03}:   '
                                             f'J={history["Objective"][-1]:0.2f}   '
                                             f'V={history["Volume"][-1]:0.2f}   '
                                             f'Conv={history["Convergence"][-1]:<0.5f}')


def optimize(optimizer, plot=False):
    p = Plotter(optimizer.x)
    print('Optimization started.')
    print("Epoch   Objective   Volume    Convergence Criteria")
    ro = optimizer.optimize(lambda i, x, history: iter_print(i, x, history, p if plot else None))
    print(f'Model converged in {(optimizer.history["Time"]):0.2f} seconds.')
    p.show_plots(optimizer.history)
    p.show(ro, interactive=False)
