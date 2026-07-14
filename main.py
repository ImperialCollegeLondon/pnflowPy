from datetime import date
import sys
import os
import dill
import joblib

from pnflowPy.inputData import InputData
from pnflowPy.network import Network
import pnflowPy.sPhase as sPhase
import pnflowPy.tPhaseD as tPhaseD
import pnflowPy.tPhaseImb as tPhaseImb
import pnflowPy.utilities as do
from pnflowPy.tPhaseD import TwoPhaseDrainage as PDrainage
from pnflowPy.SecondaryProcesses import SecDrainage, SecImbibition
from pnflowPy.tPhaseImb import TwoPhaseImbibition as PImbibition


sys.path.append("./pnflowPy")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# __DATE__ = "Jul 25 , 2023"
__DATE__ = date.today().strftime("%b") + " " + str(date.today().day) + ", " +\
      str(date.today().year)
MEMORY_DIR = f"quasi_static_results/"

def main():
    try:
        input_file_name = ""

        print("\nNetwork Model Code version 2 alpha, built: ", __DATE__, "\n")
    
        if len(sys.argv) > 1:
            input_file_name = sys.argv[1]
        else:
            input_file_name = input("Please input data file : ")

        input_data = InputData(input_file_name)
        netsim = Network(input_file_name)

        # Single Phase computation
        sPhase.initialize(netsim)
        sPhase.singlephase(netsim)

        writeData = True
        fillTillNWDisconnected = True
        
        state_data = input_data.loadState('LOAD_INIT_NETWORK_STATE')
        if state_data[0]=='T':
            file_path = state_data[1]
            try:
                loaded_obj = joblib.load(file_path)
                netsim.satW = loaded_obj['satW']
            except Exception as exc:
                print("\n\n Exception on processing of loaded state: \n", exc, "Aborting!\n")
                return 1
        
        # two Phase simulations
        if input_data.satControl():
            firstDrainCycle = True
            firstImbCycle = True
            firstCycle = True
            netsim.cycle = 0
            
            for j in range(len(input_data.satControl())):
                netsim.finalSat, Pc, netsim.dSw, netsim.minDeltaPc,\
                    netsim.deltaPcFraction, netsim.calcKr, netsim.calcI,\
                    netsim.InjectFromLeft, netsim.InjectFromRight,\
                    netsim.EscapeFromLeft, netsim.EscapeFromRight =\
                    input_data.satControl()[j]
                netsim.filling = True

                if netsim.finalSat < netsim.satW:
                    # Drainage process
                    netsim.is_oil_inj = True
                    if firstDrainCycle:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, 
                            netsim.delta, netsim.eta, netsim.distModel,
                            netsim.sepAng, netsim.CAFile) = input_data.initConAng('INIT_CONT_ANG')
                        PDrainage(netsim, writeData=writeData)

                        tPhaseD.initialize(netsim)
                        netsim.prop_drainage = {}
                        netsim.prop_drainage['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_drainage['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_drainage['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        firstDrainCycle = False
                    else:
                        SecDrainage(netsim, writeData=writeData)


                    if firstCycle and state_data[0]=='T':
                        do.updateObj(netsim, loaded_obj)                        
                        netsim.capPresMax = netsim.Pc
                        
                    netsim.maxPc = Pc
                    tPhaseD.drainage(netsim)

                else:
                    # Imbibition process
                    netsim.is_oil_inj = False

                    if firstImbCycle:
                        (netsim.wettClass, netsim.minthetai, netsim.maxthetai, netsim.delta,
                            netsim.eta, netsim.distModel, netsim.sepAng, netsim.CAFile) = input_data.initConAng('EQUIL_CON_ANG')
                                
                        PImbibition(netsim, writeData=writeData)
                        tPhaseImb.initialize(netsim)
                        netsim.prop_imbibition = {}
                        netsim.prop_imbibition['contactAng'] = netsim.contactAng.copy()
                        netsim.prop_imbibition['thetaRecAng'] = netsim.thetaRecAng.copy()
                        netsim.prop_imbibition['thetaAdvAng'] = netsim.thetaAdvAng.copy()
                        firstImbCycle = False
                    else:
                        SecImbibition(netsim, writeData=writeData)


                    if firstCycle and state_data[0]=='T':
                        do.updateObj(netsim, loaded_obj)                        
                        netsim.capPresMin = netsim.Pc
                        
                    netsim.minPc = Pc
                    netsim.fillTillNWDisconnected = fillTillNWDisconnected
                    tPhaseImb.imbibition(netsim)

                firstCycle = False

    except Exception as exc:
        print("\n\n Exception on processing: \n", exc, "Aborting!\n")
        return 1
    except:
        print("\n\n Unknown exception! Aborting!\n")
        return 1

    return 0


def load_file(file_path, netsim):
    try:
        loaded_obj = joblib.load(file_path)
        do.updateObj(netsim, loaded_obj)
        write_drainage_result(netsim)
        return 0
    except Exception as exc:
        print("\n\n Exception on processing of loaded state: \n", exc, "Aborting!\n")
        return 1


def write_drainage_result(self):
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Drainage Cycle {}---------------------'.format(self.cycle))
    print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
    %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
    self.satW, self.qW, self.krw, self.qNW, self.krnw, self.capPresMax, self.totNumFill, ))
    print('\n\n')


def write_imbibition_result(self):
    print('----------------------------------------------------------------------------------')
    print('---------------------------------Two Phase Imbibition Cycle {}---------------------'.format(self.cycle))
    print('Sw: %10.6g  \tqW: %8.6e  \tkrw: %12.6g  \tqNW: %8.6e  \tkrnw:\
    %12.6g  \tPc: %8.6g\t %8.0f invasions' % (
    self.satW, self.qW, self.krw, self.qNW, self.krnw, self.capPresMin, self.totNumFill, ))
    print('\n\n')


if __name__ == "__main__":
    sys.exit(main())


