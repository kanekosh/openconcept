from __future__ import division
import numpy as np
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
from openmdao.api import Group, IndepVarComp, ExplicitComponent

from openmdao.api import Problem, IndepVarComp

import pickle
import os


class TurbofanPropulsionSystem(Group):
    """
    Turbofan engine model using pyCycle-trained surrogate

    Inputs: fltcond|M, fltcond|h, throttle
    Outputs: thrust [lbf], fuel_flow [lbm/h] (all engines total)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        # compute thrust and SFC
        self.add_subsystem('thrustmodel', TurbofanThrustSurrogate(num_nodes=nn), 
                            promotes_inputs=['*'], promotes_outputs=['thrust'])
        self.add_subsystem('SFCmodel', TurbofanSFCSurrogate(num_nodes=nn), 
                            promotes_inputs=['*'], promotes_outputs=None)
        # compute fuelflow = thrust * SFC
        fuelflow = ElementMultiplyDivideComp()
        fuelflow.add_equation('fuel_flow', ['thrust', 'SFC'], input_units=['lbf', 'lbm / h / lbf'], vec_size=nn)
        self.add_subsystem('fuel_flow', fuelflow, promotes_outputs=['fuel_flow'])
        self.connect('thrust', 'fuel_flow.thrust')
        self.connect('SFCmodel.SFC', 'fuel_flow.SFC')

class TurbofanThrustSurrogate(ExplicitComponent):
    """
    Turbofan thrust model based on pyCycle-trained surrogate.
    Surrogate model should take input = [Mach, alt(kft), throttle] in this order. 

    Inputs: flight conditions (Mach, altitude), throttle, number of engines.  (surrogate model thrust is per engine.)
    Outputs: thrust
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('throttle', shape=(nn,), desc='Throttle (0-1)')
        self.add_input('fltcond|M', shape=(nn,), desc='Mach number')
        self.add_input('fltcond|h', units='kft', shape=(nn,), desc='altitude in kilo feet')
        self.add_input('ac|propulsion|num_engine', val=2, desc='Number of engines. Default: twin')  
        self.add_output('thrust', units='lbf', shape=(nn,), desc='thrust produced')
        self.declare_partials(['thrust'], ['throttle', 'fltcond|M', 'fltcond|h'], method='exact', rows=arange, cols=arange)
        # self.declare_partials(['thrust'], ['throttle', 'fltcond|M', 'fltcond|h'], method='fd')


        # setup surrogate
        surrogate_file = os.path.dirname(__file__) + '/surrogate_models/CFM56_thrust.pkl'
        with open(surrogate_file, 'rb') as file:
            surrogate_model = pickle.load(file)
        self.surrogate_prediction = surrogate_model['surrogate_prediction']
        self.surrogate_derivative = surrogate_model['surrogate_derivative']
        self.surrogate_scaler = surrogate_model['output_scaling_factor']
    
    def compute(self, inputs, outputs):
        # thrust at given Mach, altitude, and throttle
        x = np.vstack([inputs['fltcond|M'], inputs['fltcond|h'], inputs['throttle']]).T  # (nn * 3)
        thrust_scaled = self.surrogate_prediction(x)[:, 0]
        # unscale
        thrust_unscaled = self.surrogate_scaler[0] + thrust_scaled * (self.surrogate_scaler[1] - self.surrogate_scaler[0])
        # multiply number of engines
        outputs['thrust'] = thrust_unscaled * inputs['ac|propulsion|num_engine']

    def compute_partials(self, inputs, J):
        x = np.vstack([inputs['fltcond|M'], inputs['fltcond|h'], inputs['throttle']]).T  # (nn * 3)
        # unscale by (self.surrogate_scaler[1] - self.surrogate_scaler[0])
        unscale = self.surrogate_scaler[1] - self.surrogate_scaler[0]
        J['thrust', 'fltcond|M'] = self.surrogate_derivative(x, kx=0)[:, 0] * unscale * inputs['ac|propulsion|num_engine']
        J['thrust', 'fltcond|h'] = self.surrogate_derivative(x, kx=1)[:, 0] * unscale * inputs['ac|propulsion|num_engine']
        J['thrust', 'throttle'] = self.surrogate_derivative(x, kx=2)[:, 0] * unscale * inputs['ac|propulsion|num_engine']

class TurbofanSFCSurrogate(ExplicitComponent):
    """
    Turbofan SFC model based on pyCycle-trained surrogate.
    Surrogate model should take input = [Mach, alt(kft), throttle] in this order. 

    Inputs: flight conditions (Mach, temperature)
    Outputs: SFC: thrust-specific fuel comsumption
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('throttle', shape=(nn,), desc='Throttle (0-1)')
        self.add_input('fltcond|M', shape=(nn,), desc='Mach number')
        self.add_input('fltcond|h', units='kft', shape=(nn,), desc='altitude')
        self.add_output('SFC', units='lbm/(h * lbf)', shape=(nn,), desc='thrust-specific fuel comsumption, lbm/hour/lbf')
        self.declare_partials(['SFC'], ['throttle', 'fltcond|M', 'fltcond|h'], method='exact', rows=arange, cols=arange)
        # self.declare_partials(['SFC'], ['throttle', 'fltcond|M', 'fltcond|h'], method='fd')

        # setup surrogate
        surrogate_file = os.path.dirname(__file__) + '/surrogate_models/CFM56_tsfc.pkl'
        with open(surrogate_file, 'rb') as file:
            surrogate_model = pickle.load(file)
        self.surrogate_prediction = surrogate_model['surrogate_prediction']
        self.surrogate_derivative = surrogate_model['surrogate_derivative']
        self.surrogate_scaler = surrogate_model['output_scaling_factor']

    def compute(self, inputs, outputs):
        # SFC at given Mach, altitude, and throttle
        x = np.vstack([inputs['fltcond|M'], inputs['fltcond|h'], inputs['throttle']]).T  # (nn * 3)
        sfc_scaled = self.surrogate_prediction(x)[:, 0]
        # unscale
        outputs['SFC'] = self.surrogate_scaler[0] + sfc_scaled * (self.surrogate_scaler[1] - self.surrogate_scaler[0])

    def compute_partials(self, inputs, J):
        x = np.vstack([inputs['fltcond|M'], inputs['fltcond|h'], inputs['throttle']]).T  # (nn * 3)
        unscale = (self.surrogate_scaler[1] - self.surrogate_scaler[0])
        J['SFC', 'fltcond|M'] = self.surrogate_derivative(x, kx=0)[:, 0] * unscale
        J['SFC', 'fltcond|h'] = self.surrogate_derivative(x, kx=1)[:, 0] * unscale
        J['SFC', 'throttle'] = self.surrogate_derivative(x, kx=2)[:, 0] * unscale

if __name__ == '__main__':
    prob = Problem()
    # add input M, alt, throttle
    inv = prob.model.add_subsystem('inv', IndepVarComp(), promotes_outputs=['*'])
    inv.add_output('fltcond|M', np.array([0.8, 0.8, 0.001])) 
    inv.add_output('fltcond|h', np.array([33000, 33000, 0]), units='ft')
    inv.add_output('throttle', np.array([0.,-0.1, 1.0])) 

    prob.model.add_subsystem('engine', TurbofanPropulsionSystem(num_nodes=3), promotes_inputs=['*'])

    prob.setup()
    prob.check_partials(compact_print=True, step=1.e-7)
    prob.run_model()

    print('per engine')
    print('thrust [lbf]:', prob['engine.thrust']/2)
    print('fuel flow [lbm/h]', prob['engine.fuel_flow']/2)
    print('SFC [lbm/lbf/h]', prob['engine.SFCmodel.SFC'])