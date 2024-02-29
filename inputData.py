import numpy as np
from time import time
import os


class InputData:
    DUMMY_INDEX = -99

    def __init__(self, inputFile):
        self.workingSatEntry = 0
        self.numInletThroats = 0
        self.averageThroatLength = 0.0
        self.networkSeparation = 0.0
        self.connectionsRemoved = 0
        self.useAvrXOverThroatLen = False
        self.addPeriodicBC = False
        self.useAvrPbcThroatLen = False
        self.data = self.inputdata(inputFile)
        self.cwd = os.path.dirname(inputFile)
        
        
    def inputdata(self, inputFile):
        lines = iter(list(filter(None, (
            line.partition("%")[0].rstrip() for line in open(inputFile)))))
        data = {}
        for i in lines:
            while ';' not in i:
                i = i+next(lines)
            i = i.rstrip(";").split()
            key, value = i[0], []
            for j in i[1:]:
                try:
                    value.append(int(j))
                except ValueError:
                    try:
                        value.append(float(j))
                    except ValueError:
                        value.append(j)
            data[key] = value[0] if len(value) == 1 else value
        
        return data

    def network(self):
        if self.data['NETWORK']:
            return self.data['NETWORK'][1]
        elif self.data['TITLE']:
            return self.data['TITLE']
        else:
            print('\nError: Network name was not stated!!')
            exit()
        
    def randSeed(self):
        if self.data["RAND_SEED"]:
            return self.data["RAND_SEED"]
        else:
            return int(time.time())

    def __calcBox__(self):
        if self.data['CALC_BOX']:
            return self.data['CALC_BOX']
        else:
            return [0.5, 1.0]

    def satControl(self):
        if self.data["SAT_CONTROL"]:
            if len(self.data["SAT_CONTROL"]) % 11 == 0:
                satControl = []
                a = 0
                while len(self.data["SAT_CONTROL"]) > a:
                    satControl.append(self.data["SAT_CONTROL"][a:a+11])
                    a += 11
                
                data = np.array(satControl)[:, 0].astype('float')
                if ((data < 0) | (data > 1)).any():
                    print("\nError: Saturations to be given as fractions")
                    exit()

                return satControl
            else:
                print("\nError: Invalid entry for the SAT_CONTROL keyword")
                exit()
        else:
            return False

    def satCovergence(self):
        if self.data["SAT_COVERGENCE"]:
            return self.data["SAT_COVERGENCE"]
        else:
            return [10, 0.02, 0.8, 2.0, 'F']

    def initConAng(self, case):
        if case == 'INIT_CONT_ANG':
            data = np.array(self.data['INIT_CONT_ANG'])
            wettClass = data[0].astype('int')
            minAng, maxAng = data[1:3].astype('float')*np.arccos(-1.0)/180
            delta, eta = data[3:5].astype('float')
            try:
                distModel = data[5]
                sepAng = data[6].astype('float')*np.arccos(-1.0)/180
            except ValueError:
                distModel = 'rand'
                sepAng = 25.2*np.arccos(-1.0)/180
        elif case == 'EQUIL_CON_ANG':
            data = np.array(self.data['EQUIL_CON_ANG'])
            wettClass = data[0].astype('int')
            minAng, maxAng = data[1:3].astype('float')*np.arccos(-1.0)/180
            delta, eta = data[3:5].astype('float')
            try:
                distModel = data[5]
                sepAng = data[6].astype('float')*np.arccos(-1.0)/180
            except ValueError:
                distModel = 'rand'
                sepAng = 25.2*np.arccos(-1.0)/180
        else:
            print('\nError: Both keyword INIT_CONT_ANG and EQUIL_CON_ANG are \
                  missing!')
        
        return wettClass, minAng, maxAng, delta, eta, distModel, sepAng
        
    def res_format(self):
        if self.data["RES_FORMAT"]:
            res_form = self.data["RES_FORMAT"]
            matlab_format = (res_form.lower() == "matlab")
            excel_format = (res_form.lower() == "excel") or (
                res_form.lower() == "excelandmicroporosity")
            mcp_format = (res_form.lower() == "excelandmicroporosity") or (
                res_form.lower() == "upscaling")

        else:
            matlab_format = False
            excel_format = False
            mcp_format = False

        return matlab_format, excel_format, mcp_format

    def fluidproperties(self):
        if self.data["FLUID"]:
            [intfac_ten, wat_visc, oil_visc, wat_resist, oil_resist, wat_dens,
             oil_dens] = self.data["FLUID"]

            intfac_ten *= 1.0E-3    # Get it into N/m
            wat_visc *= 1.0E-3      # Get it into Pa.s
            oil_visc *= 1.0E-3      # Get it into Pa.s
        else:
            intfac_ten = 30.0E-3
            wat_visc = 1.0E-3
            oil_visc = 1.0E-3
            wat_resist = 1.0
            oil_resist = 1000.0
            wat_dens = 1000.0
            oil_dens = 1000.0

        return intfac_ten, wat_visc, oil_visc, wat_resist, oil_resist, \
            wat_dens, oil_dens

    def relPermDef(self):
        data = input("REL_PERM_DEF: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            flowRef = "single"
        
        return flowRef

    def prsBdrs(self):
        data = input("PRS_BDRS: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            return False

    def matBal(self):
        data = input("MAT_BAL: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            return False

    def apexPrs(self):
        data = input("APEX_PRS: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            return False

    def sourceNode(self):
        data = input("POINT_SOURCE: ")
        if data.strip():
            sourceNode = int(data)
            return sourceNode
        else:
            return 0

    '''def solverTune(self):
        data = input("SOLVER_TUNE: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            eps = 1.0E-15
            scaleFact = 5
            slvrOutput = 0
            condCutOff = 0.0
            verbose = False

        return [eps, scaleFact, slvrOutput, verbose, condCutOff]

    def poreFillWgt(self):
        data = input("PORE_FILL_WGT: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            weights = [0.0, 15000.0, 15000.0, 15000.0, 15000.0, 15000.0]
        
        return weights

    def poreFillAlg(self):
        data = input("PORE_FILL_ALG: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            algorithm = "blunt2"
        
        return algorithm

    def getModifyRadDistOptions(self):
        data = input("MODIFY_RAD_DIST: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            throatModel = 0
            poreModel = 0
            throatOptions = ""
            poreOptions = ""
            maintainLtoR = False
            writeDistToFile = False
            numPtsRDist = 0

    def getModifyGDist(self):
        data = input("MODIFY_G_DIST: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            throatModel = 0
            poreModel = 0
            throatOptions = ""
            poreOptions = ""
            writeDistToFile = False
            numPts = 0

    def getModifyPoro(self):
        data = input("MODIFY_PORO: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            netPoroTrgt = 0.0
            clayPoroTrgt = 0.0

    def modifyConnNum(self):
        data = input("MODIFY_CONN_NUM: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            targetConnNum = 0.0
            model = ""

    def getModifyModelSize(self):
        data = input("MODIFY_MOD_SIZE: ")
        if data.strip():
            # Process the data as needed
            pass
        else:
            scaleFactor = 0.0
    '''