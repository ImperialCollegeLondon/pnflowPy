import pandas as pd
import numpy as np
from matplotlib import markers, axes, pyplot as plt
import os
pd.options.mode.chained_assignment = None

class makePlot():
    muw, munw, sigma = 0.821e-3, 0.838e-3, 57e-3
    lnetwork, absK, absK0 = 3.0035e-3, 2.7203e-12, 1.85e-12
    area = lnetwork**2
    lmean, rmean = 0.0001973, 2.2274e-05
    por = 0.2190

    def __init__(self, num, title,  results,
                 compWithLitData=False, compWithPrevData=False, drain=False, imbibe=False, exclude=None, include=None):

        self.colorlist = ['g', 'c', 'y', 'm', 'k', 'b', 'lightcoral', 'lime',    
                          'navy', 'tomato', 'khaki', 'olive', 'gold', 'teal', 'darkcyan', 'tan', 'limegreen']
        self.markerlist = ['v', '^', '<', '>', 'p', 'P','d', 'D', 'h', 'H', 's', 'o', 'v', '^', 
                           's', 'v', 'o', '^',  'd', 'D', 'h', 'H']
        self.linelist = ['--', ':', '-.', '--', (0, (1, 1)), (0, (5, 10)), (0, (5, 1)), 
                         (0, (3, 1, 1, 1)), (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]
        self.num = num
        self.title = title
        self.compWithLitData = compWithLitData
        self.compWithPrevData = compWithPrevData
        self.drain = drain
        self.imbibe = imbibe
        self.exclude = exclude
        self.include = include
        self.results = results
        self.img_dir = "./result_images/"
        os.makedirs(os.path.dirname(self.img_dir), exist_ok=True)

        if self.drain:
            drainageBank(self)
        elif self.imbibe:
            imbibitionBank(self)


    def formatFig(self, xlabel, ylabel, leg, xlim, ylim, loc=1):
        label_font = {'fontname': 'Arial', 'size': 14, 'color': 'black', 'weight': 'bold',
                      'labelpad': 10}
        
        legend = plt.legend(leg, frameon=False,loc=loc)
        print(legend)
        ax = plt.gca()
        ax.tick_params(direction='in', axis='both', which='major', pad=10)
        
        # Set the rotation and label coordinates for the fraction part
        #ax.yaxis.label.set_rotation(0)
        #ax.yaxis.set_label_coords(-0.19, 0.5)

        # Increase font size and make it bold        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() 
                     + ax.get_yticklabels() + legend.get_texts()):
            item.set_fontsize(14)
            item.set_fontname('Arial')
            item.set_fontweight('bold')
            #item.set_w

        # Remove right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_xlabel(xlabel, **label_font)
        ax.set_ylabel(ylabel, **label_font, rotation=90)
        
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)

        plt.tight_layout()

            
    def pcSw(self):
        if self.drain:
            filename = self.img_dir+'Pc_vs_Sw_Drainage_{}_{}.jpg'.format(
                self.title, self.num)
        else:
            filename = self.img_dir+'Pc_vs_Sw_Imbibition_{}_{}.jpg'.format(
                self.title, self.num)
        
        leg = []
        ind = 0        
                    
        for val in self.results.keys():
            res = self.results[val]
            if val == 'Literature data':
                res = res['pcSw']
                res1 = res.loc[res['source'] == 'MICP']
                res2 = res.loc[res['source'] != 'MICP']
                if not res1.empty:
                    plt.scatter(res1['satW'], res1['capPres']/1000, s=30, marker='o',
                                facecolors='none', edgecolors='b')
                    leg.append('MICP')
                if not res2.empty:
                    plt.scatter(res2['satW'], res2['capPres']/1000, s=30, marker='s',
                                facecolors='none', edgecolors='k')
                    leg.append(val)
            elif val == 'model':
                plt.plot(res['satW'], res['capPres']/1000, '--r', linewidth=2)
                #leg.append(val)
            else:
                plt.plot(res['satW'], res['capPres']/1000, linestyle=self.linelist[ind], 
                         color=self.colorlist[ind], linewidth=2)
                leg.append(val)
                ind += 1
            

        xlabel = r'$\mathbf{S_w}$'
        ylabel = r'Capillary Pressure ($\boldsymbol{kPa}$)'
        xlim = (0, 1.01)
        ylim = (0 , 30)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()

    def krSw(self):
        if self.drain:
            filename = self.img_dir+'kr_vs_Sw_Drainage_{}_{}.jpg'.format(
                self.title, self.num)
        else:
            filename = self.img_dir+'kr_vs_Sw_Imbibition_{}_{}.jpg'.format(
                self.title, self.num)
        
        leg = []
        j = 0
        for val in self.results.keys():
            res = self.results[val]
            if val == 'Literature data':
                res = res['krSw']
                print(res)
                plt.scatter(res['satW'], res['krw'], s=30, marker='s',
                    facecolors='none', edgecolors='k')
                leg.append('Literature data (krw)')
                plt.scatter(res['satW'], res['krnw'], s=30, marker='o',
                            facecolors='none', edgecolors='b')
                leg.append('Literature data (krnw)')
                
            elif val == 'model':
                plt.plot(res['satW'], res['krw'], linestyle='--',
                        color='r', linewidth=2)
                plt.plot(res['satW'], res['krnw'], linestyle='--',
                        color='r', linewidth=2, label = '_nolegend_')
                #leg.append(val)
            
            else:
                plt.plot(res['satW'], res['krw'], linestyle=self.linelist[j], linewidth=2,
                        color=self.colorlist[j])
                plt.plot(res['satW'], res['krnw'], linestyle=self.linelist[j], linewidth=2,
                        color=self.colorlist[j], label = '_nolegend_')
                j += 1
                leg.append(val)
            
        xlabel = r'$\mathbf{S_w}$'
        ylabel = r'Relative Permeability'
        xlim = (0, 1.01)
        ylim = (0 , 1.01)
        self.formatFig(xlabel, ylabel, leg, xlim, ylim)
        plt.savefig(filename, dpi=500)
        plt.close()


class drainageBank:
    def __init__(self, obj):
        self.obj = obj
        if self.compWithLitData:
            self.__compWithLitData__()
        if self.compWithPrevData:
            self.__compWithPrevData__()

    def __getattr__(self, name):
        return getattr(self.obj, name)

    def __compWithLitData__(self):
        self.results['Literature data'] = {}
        
        self.results['Literature data']['pcSw'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Drainage_Pc_Sw.csv',
            names=['source', 'satW', 'Pc', 'capPres'], sep=',',
            skiprows=1, index_col=False)
        
        self.results['Literature data']['krSw'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Drainage_kr_Sw.csv',
            names=['satW', 'krw', 'krnw'], sep=',',
            skiprows=1, index_col=False)
        
        
        '''self.results['Valvatne et al.'] = pd.read_csv(
            './results_csv/pnflow_Bentheimer_Drainage_010725.csv',
            names=['satW', 'capPres', 'krw', 'krnw', 'RI'], sep=',',
            skiprows=1, index_col=False)'''
        
    def __compWithPrevData__(self):
        if self.include:
            todo = list(self.include)
        else:
            todo = np.arange(1, self.num).tolist()
            if self.exclude:
                todo = np.setdiff1d(todo, self.exclude).tolist()

        while True:
            try:
                n = todo.pop(0)
                self.results['model_'+str(n)] = pd.read_csv(
                    "./results_csv/FlowmodelOOP_{}_Drainage_{}.csv".format(self.title, n),
                    names=['satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                    sep=',', skiprows=18, index_col=False)
            except FileNotFoundError:
                pass
            except IndexError:
                break


class imbibitionBank():
    def __init__(self, obj):
        self.obj = obj
        if self.compWithLitData:
            self.__compWithLitData__()
        if self.compWithPrevData:
            self.__compWithPrevData__()

    def __getattr__(self, name):
        return getattr(self.obj, name)
        
    def __compWithLitData__(self):
        self.results['Literature data'] = {}
        self.results['Literature data']['pcSw'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Imbibition_Pc_Sw.csv',
            names=['source', 'satW', 'Pc', 'capPres'], sep=',',
            skiprows=1, index_col=False)
        self.results['Literature data']['krSw'] = pd.read_csv(
            './results_csv/Exp_Results_Bentheimer_Imbibition_kr_Sw.csv',
            names=['satW', 'krw', 'krnw'], sep=',',
            skiprows=1, index_col=False)
        
        '''self.results['Valvatne et al.'] = pd.read_csv(
            './results_csv/pnflow_Bentheimer_Imbibition_010725.csv', names=[
                'satW', 'capPres', 'krw', 'krnw', 'RI'], sep=',', skiprows=1,
            index_col=False)'''
            
    def __compWithPrevData__(self):
        if self.include:
            todo = list(self.include)
        else:
            todo = np.arange(1, self.num).tolist()
            if self.exclude:
                todo = np.setdiff1d(todo, self.exclude).tolist()
        while True:
            try:
                n = todo.pop(0)
                self.results['model_'+str(n)] = pd.read_csv(
                    "./results_csv/FlowmodelOOP_{}_Imbibition_{}.csv".format(self.title, n),
                    names=['satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
                    sep=',', skiprows=18, index_col=False)
            except FileNotFoundError:
                pass
            except IndexError:
                break