# -*- coding: utf-8 -*-

from matplotlib import ticker
from scipy.stats import binned_statistic_2d
from scipy.stats import gaussian_kde
from scipy.stats import sem
import copy
import matplotlib.cm as cm
import matplotlib.colors as mplcolors
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.io as pio
import seaborn as sns
pio.renderers.default = 'browser'


class _PlottingMixin:

    def plot_trajectories(self, ax, sample_num=None, color_by_time=False, color_by_random=False, DT=None, re_centre=False, save_cbar=None, fix_frames=None, **kwargs):
        
        '''
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            sample_num: sample n of the trajectories from the data...
            color_by_time: if wanting to color by the time
        
        '''

        cmap = kwargs.pop('cmap', 'viridis')
        cbar = kwargs.pop('cbar', True)
        
        cmap = plt.get_cmap(cmap)
            
        if sample_num == None:
            data_to_plot = self.data
        else:
            data_to_plot = self.data[self.data['tr'].isin(np.random.choice(self.data['tr'].unique(), sample_num))]
        
        if re_centre == True:
            data_to_plot['x'] = data_to_plot['x'] - data_to_plot.x.mean()
            data_to_plot['y'] = data_to_plot['y'] - data_to_plot.y.mean()
        
        if color_by_time:
            
            # create a cmap
            if DT == None and fix_frames == None:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min(),
                                           vmax=data_to_plot['t'].max(),
                                           clip=False)
                maximum_time = data_to_plot['t'].max()
                
            elif fix_frames != None:
                norm = mplcolors.Normalize(vmin=fix_frames[0],
                                           vmax=fix_frames[1],
                                           clip=False)
                maximum_time = fix_frames[1]
                
            else:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min()*DT,
                                           vmax=data_to_plot['t'].max()*DT,
                                           clip=False)
            
            # plot each trajectory, color by fraction within cmap
            for i in data_to_plot['tr'].unique():
                
                # getting data
                df = data_to_plot[data_to_plot['tr'] == i]
                time_fraction = df['t'].min()/maximum_time
                
                # plotting
                ax.plot(df['x'],
                        df['y'],
                        color=cmap(time_fraction),
                        **kwargs)
            
            if cbar == True:
                
                cfig = plt.figure(figsize=(0.2, 0.5))
                cax = cfig.add_axes([0.3, 0.05, 0.3, 0.9])
                cbar = cax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
                tick_locator = ticker.MaxNLocator(nbins=4)
                cbar.locator = tick_locator
                cbar.update_ticks()
                cbar.set_label('Time (frame)')
                
                if save_cbar != None:
                    cfig.savefig(save_cbar)
                    
        # plot with random colors for trajectories
        elif color_by_random:
            
            cmap_len = cmap.N
            
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df['x'],
                        df['y'],
                        color=cmap(np.random.randint(0, cmap_len)),
                        **kwargs)
        
        # plot with all trajectories of the same color
        else:
                
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df['x'],
                        df['y'],
                        **kwargs)
    
        ax.set_aspect('equal')

    def plot_trajectories_matrix(self, ax, sample_num=None, color_by_time=False, color_by_random=False, DT=None, sp_axis='x', **kwargs):
        
        '''
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            sample_num: sample n of the trajectories from the data...
            color_by_time: if wanting to color by the time
        
        '''

        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        cmap = kwargs.get('cmap', 'viridis')
        color = 'black'
        linewidth = kwargs.get('linewidth', 1)
        alpha = kwargs.get('alpha', 1)

        if sample_num == None:
            data_to_plot = self.data
        else:
            data_to_plot = self.data[self.data['tr'].isin(np.random.choice(self.data['tr'].unique(), sample_num))]
        
        if color_by_time:
            
            # create a cmap
            cmap = plt.get_cmap(cmap)
            
            if DT == None:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min(),
                                           vmax=data_to_plot['t'].max(),
                                           clip=False)
            else:
                norm = mplcolors.Normalize(vmin=data_to_plot['t'].min()*DT,
                                           vmax=data_to_plot['t'].max()*DT,
                                           clip=False)
            
            # plot each trajectory, color by fraction within cmap
            for i in data_to_plot['tr'].unique():
                
                # getting data
                df = data_to_plot[data_to_plot['tr'] == i]
                time_fraction = df['t'].min()/data_to_plot['t'].max()
                
                # plotting
                ax.plot(df[sp_axis],
                        df['t'],
                        color=cmap(time_fraction),
                        linewidth=linewidth,
                        alpha=alpha)
            
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
            cbar.set_label('Time (Frame)')
        
        # plot with random colors for trajectories
        elif color_by_random:
            
            cmap = plt.get_cmap(cmap)
            cmap_len = cmap.N
            
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df[sp_axis],
                        df['t'],
                        color=cmap(np.random.randint(0, cmap_len)),
                        linewidth=linewidth,
                        alpha=alpha)
        
        # plot with all trajectories of the same color
        else:
                
            for i in data_to_plot['tr'].unique():
                df = data_to_plot[data_to_plot['tr'] == i]
                ax.plot(df[sp_axis],
                        df['t'],
                        color=color,
                        linewidth=linewidth,
                        alpha=alpha)

    def plot_trajectories_diff_v2(self, ax, **kwargs):
        """
        function: plots trajectories using matplolib background ~ color by diffusion coefficient

        arguments:
            ax: axis to plot onto
            
        """
        
        palette = kwargs.pop('palette', 'flare')
        cmap = kwargs.pop('cmap', None)
        cbar = kwargs.pop('cbar', True)
        vmin = kwargs.pop('vmin', -5)
        vmax = kwargs.pop('vmax', 1)
        
        # get data
        data_to_plot = copy.deepcopy(self.data)
        
        # colors
        if cmap == None:
            cmap = plt.get_cmap(palette)
        
        norm = mplcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
        
        # sorting
        sorted_difflog_dict = {k: v for k, v in sorted(self.difflog_dict.items(), key=lambda item: item[1], reverse=True)}
        
        # plotting
        for tr_id, dcoef in sorted_difflog_dict.items():
            df = data_to_plot[data_to_plot['tr'] == tr_id]
            ax.plot(df['x'], df['y'], color=cmap(norm(dcoef)), **kwargs)
        
        if cbar:
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))

        ax.set_aspect('equal')

    def plot_trajectory(self, ax, tr_id, **kwargs):
        """
        function: plots trajectories using matplolib background

        arguments:
            ax: axis to plot onto
            tr_id: id's of the trajectory to plot (list)
        
        """

        # xlim = kwargs.get('xlim', None)
        # ylim = kwargs.get('ylim', None)
        # palette = kwargs.get('palette', 'flare')
        # color = kwargs.get('color', 'black')
        
        data_to_plot = self.data[self.data['tr'].isin(tr_id)]
        
        sns.lineplot(x='x', y='y', units='tr', ax=ax, data=data_to_plot, sort=False, estimator=None,
                     **kwargs)        

        # if xlim and ylim != None:
        #     ax.set_ylim(ylim)
        #     ax.set_xlim(xlim)
        
        # ax.get_legend().remove()
        ax.set_aspect('equal')

    def plot_trajectory_color_time(self, ax, tr_id, DT, **kwargs):
        
        palette = kwargs.pop('palette', 'mako')
        cmap = kwargs.pop('cmap', None)
        cbar = kwargs.pop('cbar', True)
        save_cbar = kwargs.pop('save_cbar', None)
        
        data_to_plot = copy.deepcopy(self.data[self.data['tr'] == tr_id])
        data_to_plot['t'] = data_to_plot['t'] * DT
        data_to_plot['t'] = data_to_plot['t'] - np.array(data_to_plot['t'])[0]
        
        if cmap == None:
            cmap = plt.get_cmap(palette)
            
        p1 = np.array([data_to_plot.x, data_to_plot.y]).T.reshape(-1, 1, 2)
        seg1 = np.concatenate([p1[:-1], p1[1:]], axis = 1)
        col1 = [cmap(float(i)/(len(data_to_plot)-1)) for i in range(len(data_to_plot)-1)]
        
        for ii in range(len(data_to_plot)-1):
            segii = seg1[ii]
            lii, = ax.plot(segii[:,0], segii[:,1], color = col1[ii], **kwargs)
            lii.set_solid_capstyle('round')

        ax.set_aspect('equal', 'box')
        
        if cbar:
            
            cfig = plt.figure(figsize=(0.2, 0.5))
            cax = cfig.add_axes([0.3, 0.05, 0.3, 0.9])
            norm = mplcolors.Normalize(vmin=min(data_to_plot['t']), vmax=max(data_to_plot['t']), clip=False)
            cbar = cax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
            tick_locator = ticker.MaxNLocator(nbins=4)
            cbar.locator = tick_locator
            cbar.update_ticks()
            cbar.set_label('Time (s)')
            
            if save_cbar != None:
                cfig.savefig(save_cbar)

    def plot_trajectory_color_time_3d(self, ax, tr_id, DT, **kwargs):
        
        palette = kwargs.get('palette', 'viridis')
        show_c_bar = kwargs.get('show_c_bar', True)
        linewidth = kwargs.get('linewidth', 0.3)
        
        data_to_plot = copy.deepcopy(self.data[self.data['tr'] == tr_id])
        data_to_plot['t'] = data_to_plot['t'] * DT
        data_to_plot['t'] = data_to_plot['t'] - np.array(data_to_plot['t'])[0]
        
        cmap1 = plt.get_cmap(palette)
        p1 = np.array([data_to_plot.x, data_to_plot.y, data_to_plot.t]).T.reshape(-1, 1, 3)
        seg1 = np.concatenate([p1[:-1], p1[1:]], axis = 1)
        col1 = [cmap1(float(i)/(len(data_to_plot)-1)) for i in range(len(data_to_plot)-1)]
        
        for ii in range(len(data_to_plot)-1):
            segii = seg1[ii]
            lii, = ax.plot(segii[:,0], segii[:,2], segii[:,1], color = col1[ii], linewidth = linewidth)
            lii.set_solid_capstyle('round')
        
        if show_c_bar:
            norm = mplcolors.Normalize(vmin=min(data_to_plot['t']), vmax=max(data_to_plot['t']), clip=False)
            cbar = ax.figure.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap1))
            cbar.set_label('Time (s)')

    def plot_trajectories_px(self):
        fig = px.line(x='x', y='y', color='tr', data_frame=self.data)
        fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=0,
                                                color='DarkSlateGrey'),
                                      symbol='square'),
                          selector=dict(mode='markers'))
        fig.show()

    def plot_3dtrajectories_px(self):
        fig = px.line_3d(x='x', y = 'y', z='t', color='tr', line_group='tr', data_frame=self.data)
        # fig.update_yaxes(
        #     scaleanchor = "x",
        #     scaleratio = 1)
        fig.update_traces(marker=dict(size=5,
                                      line=dict(width=0,
                                                color='DarkSlateGrey'),
                                      symbol='square'),
                          selector=dict(mode='markers'))
        fig.show()

    def plot_scatter(self, ax, centroids = False, **kwargs):
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        alpha = kwargs.get('alpha', 0.3)
        s = kwargs.get('s', 1)
        
        if centroids == False:
            sns.scatterplot(x='x', y='y', ax=ax, data=self.data, **kwargs)
        else:
            sns.scatterplot(x='x', y='y', ax=ax, data=self.centroidDF, **kwargs)
        
        if xlim and ylim != None:
            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
        
        ax.set_aspect('equal')

    def plot_scatter_kde(self, ax, **kwargs):
        
        print('estimating kde...')
        xy = np.vstack([self.data['x'], self.data['y']])
        gaus = gaussian_kde(xy)(xy)
        print('kde estimation complete...')
        
        ax.scatter(self.data['x'], self.data['y'], c=gaus, **kwargs)
        ax.set_aspect('equal')

    def plot_kde(self, ax, cbar=True, **kwargs):
        
        if cbar==True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
        
            sns.kdeplot(data=self.data, x="x", y="y", ax=ax, cbar=cbar, cbar_kws={'label': 'Probability density estimate', 'format': '%.2f'}, cbar_ax=axbar, **kwargs)
            ax.set_aspect('equal')
            
            print('saving cbar_density')
            figbar.savefig('cbar_density.svg')
        
        else:
            sns.kdeplot(data=self.data, x="x", y="y", ax=ax, cbar=cbar, cbar_kws={'label': 'Probability density estimate', 'format': '%.2f'}, **kwargs)
            ax.set_aspect('equal')

    def plot_2dhist(self, ax, cbar=True, binsize=0.1, **kwargs):
        
        bins = [np.arange(self.data.x.min(), self.data.x.max(),binsize), np.arange(self.data.y.min(), self.data.y.max(),binsize)]
        
        if cbar==True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
        
            sns.histplot(data=self.data, x="x", y="y", ax=ax, bins=bins, cbar=cbar, cbar_kws={'label': 'Detections', 'format': '%.2f'}, cbar_ax=axbar, **kwargs)
            ax.set_aspect('equal')
            
            print('saving cbar_density')
            figbar.savefig('cbar_density.svg')
        
        else:
            sns.histplot(data=self.data, x="x", y="y", ax=ax, bins=bins, cbar=cbar, cbar_kws={'label': 'Detections', 'format': '%.2f'}, **kwargs)
            ax.set_aspect('equal')

    def plot_2dhist_scatter(self, ax, cbar=True, binsize=1, **kwargs):
        
        x = self.data['x'].values
        y = self.data['y'].values
        
        x_edges = np.arange(x.min(), x.max() + binsize, binsize)
        y_edges = np.arange(y.min(), y.max() + binsize, binsize)
        
        counts, x_edges, y_edges, binnumber = binned_statistic_2d(
            x, y, None, statistic='count', bins=[x_edges, y_edges]
        )
        
        x_bin = np.clip(np.digitize(x, x_edges) - 1, 0, counts.shape[0] - 1)
        y_bin = np.clip(np.digitize(y, y_edges) - 1, 0, counts.shape[1] - 1)
        point_density = counts[x_bin, y_bin]
        
        # sort ascending by density so highest-density points are drawn last (on top)
        order = np.argsort(point_density)
        x_sorted = x[order]
        y_sorted = y[order]
        density_sorted = point_density[order]

        sc = ax.scatter(x_sorted, y_sorted, c=density_sorted, **kwargs)
        ax.set_aspect('equal')
        
        if cbar == True:
            figbar = plt.figure(figsize=(0.2, 0.5))
            axbar = figbar.add_axes([0.3, 0.05, 0.3, 0.9])
            cbar = plt.colorbar(sc,cax=axbar)
            cbar.set_label('Density')
            figbar.savefig('cbar.svg') 

    def plot_centroids_kde(self, ax, bw_method, **kwargs):
        
        print('estimating kde...')
        xy = np.vstack([self.centroidDF['x'], self.centroidDF['y']])
        gaus = gaussian_kde(xy, bw_method=bw_method)(xy)
        print('kde estimation complete...')
        
        ax.scatter(self.centroidDF['x'], self.centroidDF['y'], c=gaus, **kwargs)
        ax.set_aspect('equal')

    def plot_countfreq(self, ax, **kwargs):
        '''
        plots frequency distribution for trajectory lengths
        reqires; countDF

        '''
        
        bins = kwargs.get('bins', None)
        
        if bins == None:
            ax.hist(self.countDF['Count'], bins=max(self.countDF['Count']))
        else:
            ax.hist(self.countDF['Count'], bins=bins)

        ax.text(0, 0, 'mean {}, med {}'.format(self.countDF['Count'].mean(),
                                               self.countDF['Count'].median()))
        ax.set_box_aspect(1)
        ax.set_xlabel('Count')
        ax.set_ylabel('Freq')

    def plot_centroidHull(self, ax):
        ax.plot(*self.centroidHull.exterior.xy, color='grey', alpha=0.4)

    def plot_msds(self, ax, DT = None, **kwargs):

        '''
        function; plots the msds for all trajectories within self.data (note, msds_list must have been calculated prior)
        ax; specify axis to plot on
        DT; if None, then plots with no time information, if specified float, then will these times...

        '''
        
        if DT == None:
        
            for i in self.msd_list:
                x = np.arange(0, len(i))
                ax.plot(x[1:], i[1:], **kwargs) # note, first index contains 0
                ax.set_xlabel('Time (frame)')  
        
        else:
            for i in self.msd_list:
                x = np.arange(0, len(i)) * DT
                ax.plot(x[1:], i[1:], **kwargs) # note, first index contains 0
                ax.set_xlabel('Time (s)')  
              
        ax.set_box_aspect(1)
        ax.set_ylabel(r'MSD ($\mu$m$^2$)')

    def plot_mean_msds(self, ax, plot_x_limit, DT, **kwargs):

        '''
        function; plots the mean msds for all trajectories within self.data (note, msds_list must have been calculated prior)
        ax; specify axis to plot on
        plot_x_limit; specify maximum t value to plot (in frames)
        DT; delta T for acquisition
        
        Note; degrees of freedom <= 0 for slice indicates that only one value is piped into 'sem'
        this occurs for the longest trajectory; please ignore...

        '''
        
        color = kwargs.get('color', 'black')
        label = kwargs.get('label', 'none')
        linestyle = kwargs.get('linestyle', 'solid')
        
        # length of longest list (or trajectory)
        max_length = np.max([len(i) for i in self.msd_list])

        # create vector to store mean and sem values
        mean_values = np.zeros(max_length)
        sem_values = np.zeros(max_length)

        # calculate for all... 
        for i in range(0, max_length):
            vector = []
            for msd_vals in self.msd_list:
                if i < len(msd_vals):
                    vector.append(msd_vals[i])
                
            mean_values[i] = np.mean(vector)
            sem_values[i] = sem(vector)
        
        ax.errorbar(np.arange(0, max_length)[1:plot_x_limit] * DT, mean_values[1:plot_x_limit], yerr=sem_values[1:plot_x_limit], color=color,
                    label=label, linestyle=linestyle, marker='o', markeredgecolor='white')
        ax.set_box_aspect(1)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel(r'MSD ($\mu$m$^2$)')

    def plot_surv_prob(self, ax, **kwargs):
        '''
        Plot survival probability information.

        Parameters
        ----------
        ax : matplotlib.pyplot.axis
            Axis to plot onto.

        Returns
        -------
        None.

        '''
        
        # sns.scatterplot(data=self.surv_probDF, x='trj_len_s', y='cumsum_perc',
        #                 ax=ax, **kwargs)
        
        ax.step(data=self.surv_probDF, x='trj_len_s', y='cumsum_perc', **kwargs)
        
        
        ax.set(xscale='log')
        ax.set(yscale='log')
        ax.set_ylabel('Survival probability')
        ax.set_xlabel('Time (s)')
