from __future__ import division
import sys
import os
import numpy as np

sys.path.insert(0, os.getcwd())
from openmdao.api import Problem, Group, ScipyOptimizeDriver
from openmdao.api import DirectSolver, SqliteRecorder, IndepVarComp
from openmdao.api import NewtonSolver, BoundsEnforceLS, NonlinearBlockGS, ArmijoGoldsteinLS

# imports for the airplfane model itself
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math import AddSubtractComp
from openconcept.utilities.math.integrals import Integrator
from examples.methods.costs_commuter import OperatingCost
from openconcept.utilities.dict_indepvarcomp import DictIndepVarComp
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis, AirlinerFullMissionAnalysis

# aircraft-specific
from examples.aircraft_data.A320200 import data as acdata
# from examples.propulsion_layouts.simple_turbofan import TurbofanPropulsionSystem
# from examples.propulsion_layouts.turbofan_surrogate import TurbofanPropulsionSystem
from examples.methods.weights_jettransport import JetAirlinerEmptyWeight

class A320200AirplaneModel(Group):
    """
    A custom model specific to Airbus 320-200
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']

        # no control variables other than throttle and braking 

        """
        # propulsion system 1: simple turbofan (textbook formula)
        from examples.propulsion_layouts.simple_turbofan import TurbofanPropulsionSystem
        self.add_subsystem('propmodel', TurbofanPropulsionSystem(num_nodes=nn),
                           promotes_inputs=['fltcond|*', 'ac|propulsion|max_thrust', 'throttle', 'ac|propulsion|num_engine'],
                           promotes_outputs=['thrust', 'fuel_flow'])
        """
        # propulsion system 2: pycycle surrogate
        from examples.propulsion_layouts.turbofan_surrogate import TurbofanPropulsionSystem
        self.add_subsystem('propmodel', TurbofanPropulsionSystem(num_nodes=nn),
                           promotes_inputs=['fltcond|*', 'throttle'],
                           promotes_outputs=['thrust', 'fuel_flow'])
        #"""

        # aerodynamic model
        if flight_phase not in ['v0v1', 'v1v0', 'v1vr', 'rotate']:
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'
        self.add_subsystem('drag', PolarDrag(num_nodes=nn),
                           promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source),
                                            'fltcond|q', ('e', 'ac|aero|polar|e')],
                           promotes_outputs=['drag'])

        # weight estimation models
        self.add_subsystem('OEW', JetAirlinerEmptyWeight(),
                           promotes_inputs=['ac|geom|*', 'ac|weights|*', 'ac|propulsion|*', 'ac|misc|*'],
                           promotes_outputs=['OEW'])

        # airplanes which consume fuel will need to integrate
        # fuel usage across the mission and subtract it from TOW
        nn_simpson = int((nn - 1) / 2)
        self.add_subsystem('intfuel', Integrator(num_intervals=nn_simpson, method='simpson',
                                                 quantity_units='kg', diff_units='s',
                                                 time_setup='duration'),
                           promotes_inputs=[('dqdt', 'fuel_flow'), 'duration',
                                            ('q_initial', 'fuel_used_initial')],
                           promotes_outputs=[('q', 'fuel_used'), ('q_final', 'fuel_used_final')])
        self.add_subsystem('weight', AddSubtractComp(output_name='weight',
                                                     input_names=['ac|weights|MTOW', 'fuel_used'],
                                                     units='kg', vec_size=[1, nn],
                                                     scaling_factors=[1, -1]),
                           promotes_inputs=['*'],
                           promotes_outputs=['weight'])

class A320AnalysisGroup(Group):
    """This is an example of a balanced field takeoff and three-phase mission analysis.
    """
    def setup(self):
        # Define number of analysis points to run pers mission segment
        nn = 9

        # Define a bunch of design varaiables and airplane-specific parameters
        dv_comp = self.add_subsystem('dv_comp', DictIndepVarComp(acdata, seperator='|'),
                                     promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|CLmax_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')

        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|geom|wing|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|wing|taper')
        dv_comp.add_output_from_dict('ac|geom|wing|toverc')
        dv_comp.add_output_from_dict('ac|geom|hstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|hstab|AR')
        dv_comp.add_output_from_dict('ac|geom|hstab|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|hstab|taper')
        dv_comp.add_output_from_dict('ac|geom|hstab|toverc')
        # dv_comp.add_output_from_dict('ac|geom|hstab|c4_to_wing_c4')
        dv_comp.add_output_from_dict('ac|geom|vstab|S_ref')
        dv_comp.add_output_from_dict('ac|geom|vstab|AR')
        dv_comp.add_output_from_dict('ac|geom|vstab|c4sweep')
        dv_comp.add_output_from_dict('ac|geom|vstab|taper')
        dv_comp.add_output_from_dict('ac|geom|vstab|toverc')
        # dv_comp.add_output_from_dict('ac|geom|fuselage|S_wet')
        dv_comp.add_output_from_dict('ac|geom|fuselage|width')
        dv_comp.add_output_from_dict('ac|geom|fuselage|length')
        dv_comp.add_output_from_dict('ac|geom|fuselage|height')
        # dv_comp.add_output_from_dict('ac|geom|nosegear|length')
        # dv_comp.add_output_from_dict('ac|geom|maingear|length')
        dv_comp.add_output_from_dict('ac|geom|nacelle|length')
        dv_comp.add_output_from_dict('ac|geom|nacelle|diameter')

        dv_comp.add_output_from_dict('ac|weights|MTOW')
        dv_comp.add_output_from_dict('ac|weights|W_fuel_max')
        dv_comp.add_output_from_dict('ac|weights|MLW')

        dv_comp.add_output_from_dict('ac|propulsion|max_thrust')
        dv_comp.add_output_from_dict('ac|propulsion|num_engine')

        dv_comp.add_output_from_dict('ac|misc|num_passengers_max')

        analysis = self.add_subsystem('analysis',
                                      AirlinerFullMissionAnalysis(num_nodes=nn,
                                                          aircraft_model=A320200AirplaneModel,
                                                          num_climb=4, num_descent=3), 
                                      promotes_inputs=['*'], promotes_outputs=['*'])

def run_a320_analysis():
    # Set up OpenMDAO to analyze the airplane
    num_nodes = 9
    prob = Problem()
    prob.model = A320AnalysisGroup()

    prob.model.nonlinear_solver = NewtonSolver(iprint=2)
    # prob.model.nonlinear_solver = NonlinearBlockGS(iprint=2)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 50
    prob.model.nonlinear_solver.options['atol'] = 1e-7
    prob.model.nonlinear_solver.options['rtol'] = 1e-7
    prob.model.nonlinear_solver.linesearch = ArmijoGoldsteinLS(bound_enforcement='vector', maxiter=5, print_bound_enforce=False)
    prob.setup(check=True, mode='fwd')

    # set some (required) mission parameters. Each phase needs a vertical and air-speed
    # the entire mission needs a cruise altitude and range
    # U_EAS = equivalent air speed
    # VS = vertical speed = Rate of Climb
    ones_nn = np.ones((num_nodes,))

    #"""
    # climb 1: initial climb to 5000 ft
    prob.set_val('climb1.fltcond|vs', 2500*ones_nn, units='ft/min')
    prob.set_val('climb1.fltcond|Ueas', 175*ones_nn, units='kn')
    # climb 2: 5000 to 15000 ft
    prob.set_val('climb2|h0', 5000,  units='ft')
    prob.set_val('climb2.fltcond|vs', 2000*ones_nn, units='ft/min')
    prob.set_val('climb2.fltcond|Ueas', 290*ones_nn, units='kn')
    # climb 3: 15000 to 24000 ft
    prob.set_val('climb3|h0', 15000,  units='ft')
    prob.set_val('climb3.fltcond|vs', 1400*ones_nn, units='ft/min')
    prob.set_val('climb3.fltcond|Ueas', 290*ones_nn, units='kn')
    # climb 4 (Mach climb): 24000 ft to cruise
    prob.set_val('climb4|h0', 24000,  units='ft')
    prob.set_val('climb4.fltcond|vs', 1000*ones_nn, units='ft/min')
    prob.set_val('climb4.fltcond|Ueas', 240*ones_nn, units='kn')
    # cruise. M=0.78 at 37000 ft. U_tas = 450 kn, U_eas = 240 kn
    prob.set_val('cruise|h0', 37000,  units='ft')
    prob.set_val('cruise.fltcond|vs', 0.1*ones_nn, units='ft/min')  # horizontal cruise
    prob.set_val('cruise.fltcond|Ueas', 240*ones_nn, units='kn')
    # descent 1: initial descent to 24000 ft
    prob.set_val('descent1.fltcond|vs', -500*ones_nn, units='ft/min')   # 1000
    prob.set_val('descent1.fltcond|Ueas', 240*ones_nn, units='kn')  
    # descent 2: 24000 to 10000 ft
    prob.set_val('descent2|h0', 24000,  units='ft')
    prob.set_val('descent2.fltcond|vs', -1000*ones_nn, units='ft/min')  # 3500 too steep?
    prob.set_val('descent2.fltcond|Ueas', 290*ones_nn, units='kn')
    # descent 3: approach
    prob.set_val('descent3|h0', 10000,  units='ft')
    prob.set_val('descent3.fltcond|vs', -500*ones_nn, units='ft/min')  # 1500
    prob.set_val('descent3.fltcond|Ueas', 250*ones_nn, units='kn')
    """
    prob.set_val('climb1.fltcond|vs', 1000*ones_nn, units='ft/min')
    prob.set_val('climb1.fltcond|Ueas', 70*ones_nn, units='kn')
    # cruise. M=0.78 at 37000 ft. U_tas = 450 kn, U_eas = 240 kn
    prob.set_val('cruise|h0', 20000,  units='ft')
    prob.set_val('cruise.fltcond|vs', 0.1*ones_nn, units='ft/min')  # horizontal cruise
    prob.set_val('cruise.fltcond|Ueas', 150*ones_nn, units='kn')
    # descent 1: initial descent to 24000 ft
    prob.set_val('descent1.fltcond|vs', -800*ones_nn, units='ft/min')   # 1000
    prob.set_val('descent1.fltcond|Ueas', 100*ones_nn, units='kn')  
    """

    prob.set_val('mission_range', 2200, units='NM')  # see Airbus AC-A320 pp 143 for payload-range chart

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*150,units='kn')
    prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*150,units='kn')
    prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*150,units='kn')

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    prob['v0v1.throttle'] = np.ones((num_nodes)) 
    prob['v1vr.throttle'] = np.ones((num_nodes)) 
    prob['rotate.throttle'] = np.ones((num_nodes)) 

    prob.run_model()
    return prob

if __name__ == "__main__":
    from openconcept.utilities.visualization import plot_trajectory
    # run the analysis
    prob = run_a320_analysis()

     # print some outputs
    vars_list = ['ac|weights|MTOW','climb1.OEW','descent3.fuel_used_final','rotate.range_final','engineoutclimb.gamma']
    units = ['lb','lb','lb','ft','deg']
    nice_print_names = ['MTOW', 'OEW', 'Fuel used', 'TOFL (over 35ft obstacle)','Climb angle at V2']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])

    # plot some stuff
    plots = True
    if plots:
        x_var = 'range'
        x_unit = 'ft'
        y_vars = ['fltcond|Ueas', 'fltcond|h']
        y_units = ['kn', 'ft']
        x_label = 'Distance (ft)'
        y_labels = ['Veas airspeed (knots)', 'Altitude (ft)']
        phases = ['v0v1', 'v1vr', 'rotate', 'v1v0']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels,
                        plot_title='A320-200 Takeoff')

        x_var = 'range'
        x_unit = 'NM'
        y_vars = ['fltcond|h','fltcond|Ueas','fuel_used','throttle','fltcond|vs', 'fltcond|M']
        # y_units = ['ft','kn','lbm',None,'ft/min', None]
        y_units = ['ft','kn','lbm',None,'ft/min', None]
        x_label = 'Range (nmi)'
        y_labels = ['Altitude (ft)', 'Veas airspeed (knots)', 'Fuel used (lb)', 'Throttle setting', 'Vertical speed (ft/min)', 'Mach']
        # phases = ['climb1', 'climb2', 'cruise', 'descent1', 'descent2']
        # phases = ['climb1', 'cruise', 'descent1']
        phases = ['climb1', 'climb2', 'climb3', 'climb4', 'cruise', 'descent1', 'descent2', 'descent3']
        plot_trajectory(prob, x_var, x_unit, y_vars, y_units, phases,
                        x_label=x_label, y_labels=y_labels, marker='-',
                        plot_title='A320-200 Mission Profile')

        plot_trajectory(prob, 'fltcond|M', None, ['fltcond|h'], ['ft'], phases,
                        x_label='Mach', y_labels=['altitude [ft]'], marker='-',
                        plot_title='A320-200 Mission: Mach - altitude')
