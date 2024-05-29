import matplotlib.colors as mc
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


class Plotter:

    def __init__(self, x):
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(6, 6))

        self.im = self.ax.imshow(x, origin='lower', cmap='Greys', vmin=0, vmax=1, interpolation='nearest')

        self.fig.canvas.manager.set_window_title('Optimization result')
        self.ax.title.set_text('Initializing')
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.im)
        self.ax.tick_params(axis='both', which='both', colors='w')
        self.fig.canvas.blit(self.fig.bbox)
        plt.draw()
        plt.pause(0.1)

    def show(self, x, title=None, interactive=False):
        array = x.copy()
        self.fig.canvas.restore_region(self.bg)
        self.im.set_array(array)
        self.ax.draw_artist(self.im)
        if title is not None: self.ax.title.set_text(title)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
        if not interactive: plt.show(block=True)

    @staticmethod
    def show_plots(history):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        fig.canvas.manager.set_window_title('Convergence history')
        ax[0].plot(history['Objective'] / max(history['Objective']), label='Objective')
        ax[0].title.set_text('Objective')
        ax[1].plot(history['Volume'], label='Volume')
        ax[1].plot(history['Convergence'], label='Convergence Criteria')
        ax[1].title.set_text('Constraints')
        plt.ylim(0, 1)
        ax[1].legend()
        plt.show()
