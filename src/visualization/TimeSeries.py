import matplotlib.pyplot as plt
from os.path import join

class TimeSeries:
    colors = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']
    view_results = False
    save_figures = False
    output_folder = ''

    def __init__(self, data, disp=True, save_fig=False, output_folder=''):
        self.view_results = disp
        self.save_figures = save_fig
        self.data = data
        self.output_folder = output_folder

    def setDisplay(self,disp):
        self.view_results = disp

    def dispImages(self):
        ''' Used to decide if we want to show the results or not (normally just save)'''
        if self.view_results:
            plt.show()
        else:
            plt.close()

    def saveFigure(self, title):
        '''Just test if we need to save the figure, if true then we save it'''
        if self.save_figures:
            plt.savefig(join(self.output_folder, title), bbox_inches='tight')

    def plotDfColumnsWithScatter(self, columns, xcol='', savefig='', title='', legends='', method='single'):
        '''
        Plots a series of columns inside a data frame. It can be in separate plots o in just one
        :param columns: which columns to plot
        :param xcol: Which column we want to use for the x axis
        :param savefig: Name of the file to use for saving
        :param title: Title of the plot
        :param legends: Legends for each column
        :param method: 'single': Separated windows 'merged': all in same figure  'subplot' Single image multiple plots
        :return:
        '''
        fig_size = 8
        tot_columns = len(columns)

        # Define the figure depending on the mode
        if method == 'merged':
            plt.figure(figsize=(fig_size, fig_size))
        if method == 'subplot':
            plt.figure(figsize=(fig_size*tot_columns, fig_size))

        # Iterate over each column
        for ii, cur_col_name in enumerate(columns):
            try:
                if method == 'single':
                    plt.figure(figsize=(fig_size, fig_size))
                if method == 'subplot':
                    plt.subplot(1,tot_columns,ii+1)

                plt.title(cur_col_name)

                cur_col = self.data[cur_col_name]
                # Choose what are we using for X axis
                if xcol != '':
                    if xcol == 'index':
                        X = self.data.index
                    else:
                        X = self.data[xcol]
                else:
                    X = range(len(cur_col.keys()))

                if legends != '':
                    plt.scatter(X, cur_col.values, label=legends[ii])
                else:
                    plt.scatter(X, cur_col.values)

                if method == 'single':
                    if title != '':
                        plt.title(title)
                    self.saveFigure(F'{cur_col_name}_{title}')
                    self.dispImages()
            except Exception as e:
                print("----- Failed to make scatter plot: ", e)

            if method == 'subplot' or  method == 'merged':
                if method == 'merged':
                    plt.legend(loc='best')
                self.saveFigure(F'All_{title}')
                self.dispImages()


