import pandas as pd
from plot import makePlot

#title = 'Bentheimer'
title = 'estaillades_sublabel2'
title = 'res_sample'
num = 1

drainage_results = {}
'''drainage_results['model'] = pd.read_csv(
    './results_csv/FlowmodelOOP_Bentheimer_Drainage_010725.csv', names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)'''
drainage_results['model'] = pd.read_csv(
    './results_csv/Flowmodel_{}_Drainage_{}.csv'.format(
        title, num), names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)

imbibition_results = {}
imbibition_results['model'] = pd.read_csv(
    './results_csv/Flowmodel_{}_Imbibition_{}.csv'.format(
        title, num), names=[
    'satW', 'qWout', 'krw', 'qNWout', 'krnw', 'capPres', 'invasions'],
    sep=',', skiprows=18, index_col=False)


drain = True
imbibe = True

if drain:
    #mkD = makePlot(num, title, drainage_results, True, True, True, False, include=None)
    mkD = makePlot(num, title, drainage_results, drain=True)
    mkD.pcSw()
    mkD.krSw()
if imbibe:
    #mkI = makePlot(num, title, imbibition_results, True, True, False, True, include=None)
    mkI = makePlot(num, title, imbibition_results, imbibe=True)
    mkI.pcSw()
    mkI.krSw()