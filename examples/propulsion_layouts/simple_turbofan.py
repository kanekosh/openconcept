from __future__ import division
import numpy as np
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
from openmdao.api import Group, IndepVarComp, ExplicitComponent

from openmdao.api import Problem, IndepVarComp

class TurbofanPropulsionSystem(Group):
    """
    Inputs: fltcond|M, fltcond|T, fltcond|rho, ac|propulsion|max_thrust, throttle
    Outputs: thrust [lbf], fuel_flow [lbm/h] (all engines total)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        # compute thrust and SFC
        self.add_subsystem('thrustmodel', TurbofanSimpleThrust(num_nodes=nn), 
                            promotes_inputs=['*'], promotes_outputs=['thrust'])
        self.add_subsystem('SFCmodel', TurbofanSimpleSFC(num_nodes=nn), 
                            promotes_inputs=['*'], promotes_outputs=None)
        # compute fuelflow = thrust * SFC
        fuelflow = ElementMultiplyDivideComp()
        fuelflow.add_equation('fuel_flow', ['thrust', 'SFC'], input_units=['lbf', 'lbm / h / lbf'], vec_size=nn)
        self.add_subsystem('fuel_flow', fuelflow, promotes_outputs=['fuel_flow'])
        self.connect('thrust', 'fuel_flow.thrust')
        self.connect('SFCmodel.SFC', 'fuel_flow.SFC')


class TurbofanSimpleThrust(ExplicitComponent):
    """
    simple turbofan engine
    Inputs: flight conditions (Mach, rho), throttle, thrust_sls_max
    Outputs: thrust

    Assuming Thrust_sls is linear w.r.t. throttle (T_sls = throttle * T_sls_max)
        then compute thrust = f(T_sls, Mach, altitude)
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('throttle', shape=(nn,), desc='Throttle (0-1)')
        # self.add_input('fltcond|M', shape=(nn,), desc='Mach number')
        self.add_input('fltcond|rho', units='kg/m**3', shape=(nn,), desc='Air density')
        self.add_input('ac|propulsion|max_thrust', units='lbf', desc='max thrust at sea level static (n engines total)')
        self.add_input('ac|propulsion|num_engine', val=2, desc='Number of engines. Default: twin')  
        self.add_output('thrust', units='lbf', shape=(nn,), desc='thrust produced')
        self.declare_partials(['thrust'], ['throttle', 'fltcond|rho'], method='exact', rows=arange, cols=arange)
        self.declare_partials(['thrust'], ['ac|propulsion|max_thrust'], method='exact', rows=arange, cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        # assume linear relation between thrust_sls and throttle
        num_engine = inputs['ac|propulsion|num_engine']
        T_sls = inputs['throttle'] * inputs['ac|propulsion|max_thrust'] 
        # thrust at sea level, given Mach
        T_sl_m = T_sls  # thrust-Mach dependence is ignored. not physical, but implicit solver for throttle can handle this. 
        # thrust at given air density & Mach. hardcoding rho0 = 1.225 kg/m**3, 
        outputs['thrust'] = T_sl_m * (inputs['fltcond|rho'] / 1.225)**1. * num_engine

    def compute_partials(self, inputs, J):
        num_engine = inputs['ac|propulsion|num_engine']
        J['thrust', 'throttle'] = inputs['ac|propulsion|max_thrust'] * (inputs['fltcond|rho'] / 1.225)**1. * num_engine
        J['thrust', 'ac|propulsion|max_thrust'] = inputs['throttle'] * (inputs['fltcond|rho'] / 1.225)**1. * num_engine
        J['thrust', 'fltcond|rho'] = inputs['ac|propulsion|max_thrust'] * inputs['throttle'] / 1.225 * num_engine


class TurbofanSimpleSFC(ExplicitComponent):
    """
    simple SFC model 
    Inputs: flight conditions (Mach, temperature)
    Outputs: SFC: thrust-specific fuel comsumption
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']
        arange = np.arange(0, nn)
        self.add_input('fltcond|M', shape=(nn,), desc='Mach number')
        self.add_input('fltcond|T', units='K', shape=(nn,), desc='Temperature')
        self.add_output('SFC', units='lbm/(h * lbf)', shape=(nn,), desc='thrust-specific fuel comsumption, lbm/hour/lbf')
        self.declare_partials(['SFC'], ['*'], method='exact', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # source: mattingly, 1996, Eq. (1.36a)
        theta = inputs['fltcond|T'] / 288.16
        sfc = (0.4 + 0.45 * inputs['fltcond|M']) * np.sqrt(theta)
        outputs['SFC'] = sfc

    def compute_partials(self, inputs, J):
        theta = inputs['fltcond|T'] / 288.16
        J['SFC', 'fltcond|M'] = 0.45 * np.sqrt(theta)
        J['SFC', 'fltcond|T'] = (0.4 + 0.45 * inputs['fltcond|M']) / (2 * 288.16**0.5) * inputs['fltcond|T']**(-0.5)
