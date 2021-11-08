import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp, BalanceComp, ExecComp
import openconcept.api as oc
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.analysis.aerodynamics import Lift
from openconcept.utilities.math.integrals import Integrator
from openconcept.analysis.performance.solver_phases import VerticalAcceleration, Groundspeeds, SteadyFlightCL, HorizontalAcceleration


class VerticalAccelerationTiltRotor(ExplicitComponent):
    """
    Computes vertical acceleration with tilted rotor
    
    Inputs
    ------
    weight : float
        Aircraft weight (scalar, kg)
    drag : float
        Aircraft drag at each analysis point (vector, N)
    lift : float
        Aircraft lift at each analysis point (vector, N)
    thrust : float
        Thrust at each TO analysis point (vector, N)
    fltcond|singamma : float
        The sine of the flight path angle gamma (vector, dimensionless)
    fltcond|cosgamma : float
        The cosine of the flight path angle gamma (vector, dimensionless)
    Tangle : float
        Rotor tilt angle angle w.r.t. horizontal plane (vector, rad)

    Outputs
    -------
    accel_vert : float
        Aircraft horizontal acceleration (vector, m/s**2)

    Options
    -------
    num_nodes : int
        Number of analysis points to run
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('weight', units='kg', shape=(nn,))
        self.add_input('drag', units='N', shape=(nn,))
        self.add_input('lift', units='N', shape=(nn,))
        self.add_input('thrust', units='N', shape=(nn,))
        self.add_input('fltcond|singamma', shape=(nn,))
        self.add_input('fltcond|cosgamma', shape=(nn,))
        self.add_input('Tangle', units='rad', shape=(nn,))

        self.add_output('accel_vert', units='m/s**2', shape=(nn,))
        arange = np.arange(nn)
        self.declare_partials('*', '*', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        g = 9.80665  # m/s^2
        m = inputs['weight']
        cosg = inputs['fltcond|cosgamma']
        sing = inputs['fltcond|singamma']
        sint = np.sin(inputs['Tangle'])
        
        # vertical EoM is: m vaccel = L cos(gamma) + T sin(Tangle) - D sin(gamma) - mg
        accel = inputs['lift'] * cosg / m + inputs['thrust'] * sint / m - inputs['drag'] * sing / m - g
        ### accel = np.clip(accel, -g, 2.5*g)
        outputs['accel_vert'] = accel

    def compute_partials(self, inputs, J):
        m = inputs['weight']
        cosg = inputs['fltcond|cosgamma']
        sing = inputs['fltcond|singamma']
        sint = np.sin(inputs['Tangle'])
        L = inputs['lift']
        D = inputs['drag']
        T = inputs['thrust']

        J['accel_vert', 'thrust'] = sint / m
        J['accel_vert', 'drag'] = -sing / m
        J['accel_vert', 'lift'] = cosg / m
        J['accel_vert', 'fltcond|singamma'] = -D / m
        J['accel_vert', 'fltcond|cosgamma'] = L / m
        J['accel_vert', 'Tangle'] = T / m * np.cos(inputs['Tangle'])
        J['accel_vert', 'weight'] = -(L * cosg + T * sint - D * sing) / m**2


class SteadyVerticalFlightPhase(oc.PhaseGroup):
    """
    This component group models steady vertical flight conditions of VTOLs.
    Settable mission parameters include:
        - Vertical speed (fltcond|vs)
        - Duration of the segment (duration)

    Throttle is set automatically to ensure steady flight;
        i.e., thrust = drag + weight for climb, thrust + drag = weight for descent.

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs of aircraft model
    ------
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)

    Outputs of aircraft model
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None, desc='Phase of flight. currently only supports: climb, hover, descent')
        self.options.declare('aircraft_model', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']

        # constant parameters of the flight phase. Set (overwrite) these values from runscript using prob.set_val()
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output('propulsor_active', val=np.ones(nn))
        ivcomp.add_output('zero_accel', val=np.zeros((nn,)), units='m/s**2')   # later used to solve for throttle.
        ivcomp.add_output('dt_dt', val=np.ones(nn), units='s/s')  # dt/dt = 1. Will integrate this to obtain the time at each flight point.

        # vertical velocity and duration of this phase.
        ivcomp.add_output('fltcond|vs', val=np.ones((nn,)) * 1, units='m/s')   # vertical speed
        ivcomp.add_output('duration', val=1., units='s')       # duration of this phase
        # for vertical flight, set lift = 0, ground speed = 0, and flight path angle = +90.
        ivcomp.add_output('lift', val=np.zeros((nn,)), units='N')
        ivcomp.add_output('fltcond|groundspeed', val=np.zeros((nn,)), units='m/s')
        ivcomp.add_output('fltcond|cosgamma', val=np.zeros((nn,)))
        ivcomp.add_output('fltcond|singamma', val=np.ones((nn,)))  # The definition of flight path angle here is the direction of thrust w.r.t. gravity. Therefore it is +90 deg for both vertical climb and descent.
        
        # integrate the vertical velocity (fltcond|vs) to compute the altitude at each flight point. Also, integrate dt/dt=1 to compute the time at each point.
        integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', time_setup='duration', method='simpson'), promotes_inputs=['fltcond|vs', 'fltcond|groundspeed', 'dt_dt'], promotes_outputs=['fltcond|h', 'range', 'mission_time'])
        integ.add_integrand('fltcond|h', rate_name='fltcond|vs', val=1.0, units='m')   # initial valut of integrand (fltcond|h) = 0 by default.
        integ.add_integrand('range', rate_name='fltcond|groundspeed', val=1.0, units='m')   # range should be 0 for this phase.
        integ.add_integrand('mission_time', rate_name='dt_dt', val=0.0, units='s')

        # atmosphere model. Connect the input "fltcond|Ueas" to 'fltcond|vs' for vertical flight.
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['fltcond|h'], promotes_outputs=['*'])
        self.connect('fltcond|vs', 'atmos.fltcond|Ueas')   # compute the dynamic pressure based on vertical speed.

        # add the user-defined aircraft model
        self.add_subsystem('acmodel', self.options['aircraft_model'](num_nodes=nn, flight_phase=self.options['flight_phase']), promotes_inputs=['*'], promotes_outputs=['*'])

        # vertical acceleration
        # the direction (sign) of the drag depends on the mission phase. Positive for climb, negative for descent.
        if flight_phase in ['climb', 'hover']:
            # The drag direction is opposite to thrust, which is as defined in VerticalAcceleration
            self.add_subsystem('vaccel', VerticalAcceleration(num_nodes=nn), promotes_inputs=['weight', 'lift', 'thrust', 'fltcond|singamma', 'fltcond|cosgamma'], promotes_outputs=['accel_vert'])
            self.connect('drag', 'vaccel.drag')
        elif flight_phase in ['descent']:
            # The drag and thrust is in the same direction. Thus flip the sign of drag.
            self.add_subsystem('sign_flipper', ExecComp('drag_minus = -1. * drag', shape=(nn,), units='N'), promotes_inputs=['drag'], promotes_outputs=['drag_minus'])
            self.add_subsystem('vaccel', VerticalAcceleration(num_nodes=nn), promotes_inputs=['weight', 'lift', 'thrust', 'fltcond|singamma', 'fltcond|cosgamma'], promotes_outputs=['accel_vert'])
            self.connect('drag_minus', 'vaccel.drag')
        else:
            raise RuntimeError('option flight_phase must be climb, hover, or descent.')
        # END IF

        # Set throttle such that vertical acceleration == 0.
        self.add_subsystem('steadyflt', BalanceComp(name='throttle', val=np.ones((nn,)) * 0.25, lower=0.001, upper=1.2, units=None, normalize=False, eq_units='m/s**2', rhs_name='accel_vert', lhs_name='zero_accel', rhs_val=np.zeros((nn,))),
                           promotes_inputs=['accel_vert', 'zero_accel'], promotes_outputs=['throttle'])


class SteadyFlightPhaseForVTOLCruise(oc.PhaseGroup):
    """
    This component group models steady flight conditions of tiltrotor VTOLs in cruise.
    Settable mission parameters include:
        - Airspeed (fltcond|Ueas)
        - Vertical speed (fltcond|vs)
        - Duration of the segment (duration)
        - Thrust tilt angle w.r.t. flight path angle (Tangle)

    Throttle is set automatically to ensure steady flight;
        - i.e., T cos(Tangle) = D + mg sin(gamma); T sin(Tangle) + L = mg cos(gamma)

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    TODO / NOTE: how can we use this component for forward flight without wings (e.g. multirotor drones)? Can we just set CL = 0 ???

    Inputs of aircraft model
    ------
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|CL : float
        Lift coefficient (vector, dimensionless)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)
    lift : float
        Lift force (vector, N)

    Outputs of aircraft model
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None, desc='Phase of flight. currently only supports: cruise')
        self.options.declare('aircraft_model', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']

        # constant parameters of the flight phase. Set (overwrite) these values from runscript using prob.set_val()
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output('propulsor_active', val=np.ones(nn))
        ivcomp.add_output('braking', val=np.zeros(nn))   # this must be 0.
        ivcomp.add_output('zero_accel', val=np.zeros((nn,)), units='m/s**2')   # later used to solve for throttle.
        ivcomp.add_output('dt_dt', val=np.ones(nn), units='s/s')  # dt/dt = 1. Will integrate this to obtain the time at each flight point.
        
        # mission parameters (settable in runscript)
        ivcomp.add_output('fltcond|Ueas', val=np.ones((nn,)) * 90, units='m/s')    # horizontal speed
        ivcomp.add_output('fltcond|vs', val=np.ones((nn,)) * 1, units='m/s')   # vertical speed
        ivcomp.add_output('duration', val=1., units='s')       # duration of this phase
        ivcomp.add_output('Tangle', val=np.ones((nn,)) * 5, units='deg')   # thrust tilt angle w.r.t. horizontal plane
        
        # integrate the vertical velocity (fltcond|vs) to compute the altitude at each flight point. Also, integrate dt/dt=1 to compute the time at each point.
        integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', time_setup='duration', method='simpson'), promotes_inputs=['fltcond|vs', 'fltcond|groundspeed', 'dt_dt'], promotes_outputs=['fltcond|h', 'range', 'mission_time'])
        integ.add_integrand('fltcond|h', rate_name='fltcond|vs', val=1.0, units='m')   # initial valut of integrand (fltcond|h) = 0 by default.
        integ.add_integrand('mission_time', rate_name='dt_dt', val=0.0, units='s')

        # atmosphere model.
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['*'], promotes_outputs=['*'])

        # compute ground speed and flight path angle
        self.add_subsystem('gs', Groundspeeds(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        integ.add_integrand('range', rate_name='fltcond|groundspeed', val=1.0, units='m')

        # add the user-defined aircraft model
        self.add_subsystem('acmodel', self.options['aircraft_model'](num_nodes=nn, flight_phase=flight_phase), promotes_inputs=['*'], promotes_outputs=['*'])

        # thrust components in the flight path and perpendicular directions
        # - flight-path projection: T cos(Tangle - gamma); cos(Tangle - gamma) = cos(Tangle) * cos(gamma) + sin(Tangle) * sin(gamma)
        # - perpendicular to the flight path: T sin(Tangle - gamma); sin(Tangle - gamma) = sin(Tangle) * cos(gamma) - cos(Tangle) * sin(gamma)
        thrust_projection = ['thrust_fp = thrust * (cos(Tangle) * cosgamma + sin(Tangle) * singamma)', 'thrust_fpnormal = thrust * (sin(Tangle) * cosgamma - cos(Tangle) * singamma)']
        thrust_dict = {'units' : 'N', 'shape' : (nn,)}
        sincos_dict = {'shape' : (nn,)}
        self.add_subsystem('thrust_proj',
                            ExecComp(thrust_projection, thrust_fp=thrust_dict, thrust_fpnormal=thrust_dict, thrust=thrust_dict, Tangle={'units' : 'rad', 'shape' : (nn,)}, singamma=sincos_dict, cosgamma=sincos_dict),
                            promotes_inputs=['thrust', 'Tangle', ('cosgamma', 'fltcond|cosgamma'), ('singamma', 'fltcond|singamma')])

        # compute the lift required (to maintain vertical acceleration = 0)
        # steady EoM perpendicular to the flight path: L + T sin(Tangle - gamma) = mg cos(gamma) . Therefore, define netweight := (m - T sin(Tangle - gamma) / (g cos(gamma))
        weight_dict = {'units' : 'kg', 'shape' : (nn,)}
        self.add_subsystem('weight_minus_thrust_fpnormal',
                            ExecComp('netweight = weight - thrust_fpnormal / 9.80665 / cosgamma', netweight=weight_dict, weight=weight_dict, thrust_fpnormal=thrust_dict, cosgamma=sincos_dict),
                            promotes_inputs=['weight', ('cosgamma', 'fltcond|cosgamma')])
        self.connect('thrust_proj.thrust_fpnormal', 'weight_minus_thrust_fpnormal.thrust_fpnormal')
        self.add_subsystem('clcomp', SteadyFlightCL(num_nodes=nn), promotes_inputs=['fltcond|q', 'ac|geom|wing|S_ref', 'fltcond|cosgamma'], promotes_outputs=['fltcond|CL'])
        self.connect('weight_minus_thrust_fpnormal.netweight', 'clcomp.weight')
        self.add_subsystem('lift', Lift(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])

        # horizontal acceleration (along the flight path)
        self.add_subsystem('haccel', HorizontalAcceleration(num_nodes=nn), promotes_inputs=['weight', 'drag', 'lift', 'fltcond|singamma', 'braking'], promotes_outputs=['accel_horiz'])
        self.connect('thrust_proj.thrust_fp', 'haccel.thrust')

        # set throttle such that horizontal acceleration = 0
        self.add_subsystem('steadyflt', BalanceComp(name='throttle', val=np.ones((nn,)) * 0.5, lower=0.001, upper=1.2, units=None, normalize=False, eq_units='m/s**2', rhs_name='accel_horiz', lhs_name='zero_accel', rhs_val=np.zeros((nn,))),
                           promotes_inputs=['accel_horiz', 'zero_accel'], promotes_outputs=['throttle'])


class UnsteadyFlightPhaseForTiltrotorTransition(oc.PhaseGroup):
    """
    This component group models transition of tiltrotor VTOLs.
    Settable mission parameters include:
        - Airspeed (fltcond|Ueas)
        - Vertical speed (fltcond|vs)
        - derivative of Ueas and vs (i.e. acceleration)
        - Duration of the segment (duration)
        - Thrust tilt angle in cruise (Tangle_cruise), for smooth transition from the transition phase to cruise

    Throttle and rotor tilt angle (theta) is set automatically to achieve the given velocity history.
        - i.e., acceleration computed = differential of the given velocity history
    CL, CD is determined given the angle of attack, where (AoA = -flight path angle)

    The BaseAircraftGroup object is passed in.
    The BaseAircraftGroup should be built to accept the following inputs
    and return the following outputs.
    The outputs should be promoted to the top level in the component.

    Inputs of aircraft model
    ------
    fltcond|h : float
        Altitude (vector, m)
    fltcond|vs : float
        Vertical speed (vector, m/s)
    fltcond|Ueas : float
        Equivalent airspeed (vector, m/s)
    fltcond|Utrue : float
        True airspeed (vector, m/s)
    fltcond|p : float
        Pressure (vector, Pa)
    fltcond|rho : float
        Density (vector, kg/m3)
    fltcond|T : float
        Temperature (vector, K)
    fltcond|q : float
        Dynamic pressure (vector, Pa)
    fltcond|alpha : float
        Angle of attack, (vector, rad)
    throttle : float
        Motor / propeller throttle setting scaled from 0 to 1 or slightly more (vector, dimensionless)
    propulsor_active : float
        If a multi-propulsor airplane, a failure condition should be modeled in the propulsion model by multiplying throttle by propulsor_active.
        It will generally be 1.0 unless a failure condition is being modeled, in which case it will be 0 (vector, dimensionless)

    Outputs of aircraft model
    -------
    thrust : float
        Total thrust force produced by all propulsors (vector, N)
    lift : float
        Lift force (vector, N)
    drag : float
        Total drag force in the airplane axis produced by all sources of drag (vector, N)
    weight : float
        Weight (mass, really) of the airplane at each point in time. (vector, kg)
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None, desc='Phase of flight. transition_climb or transition_descent')
        self.options.declare('aircraft_model', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        flight_phase = self.options['flight_phase']

        # constant parameters of the flight phase. Set (overwrite) these values from runscript using prob.set_val()
        ivcomp = self.add_subsystem('const_settings', IndepVarComp(), promotes_outputs=["*"])
        ivcomp.add_output('propulsor_active', val=np.ones(nn))
        ivcomp.add_output('braking', val=np.zeros(nn))   # this must be 0.
        ivcomp.add_output('dt_dt', val=np.ones(nn), units='s/s')  # dt/dt = 1. Will integrate this to obtain the time at each flight point.
        
        # mission parameters (settable in runscript)
        ivcomp.add_output('fltcond|Ueas', val=np.ones((nn,)) * 90, units='m/s')    # airspeed speed
        ivcomp.add_output('fltcond|vs', val=np.ones((nn,)) * 1, units='m/s')   # vertical speed
        ivcomp.add_output('duration', val=1., units='s')       # duration of this phase

        # target acceleration  TODO: differentiate Utrue and vs to compute these diffs!
        ivcomp.add_output('accel_horiz_target', val=np.zeros((nn,)), units='m/s**2')    # along the flight path
        ivcomp.add_output('accel_vert_target', val=np.zeros((nn,)), units='m/s**2')   # vertical

        # integrate the vertical velocity (fltcond|vs) to compute the altitude at each flight point. Also, integrate dt/dt=1 to compute the time at each point.
        integ = self.add_subsystem('ode_integ', Integrator(num_nodes=nn, diff_units='s', time_setup='duration', method='simpson'), promotes_inputs=['fltcond|vs', 'fltcond|groundspeed', 'dt_dt'], promotes_outputs=['fltcond|h', 'range', 'mission_time'])
        integ.add_integrand('fltcond|h', rate_name='fltcond|vs', val=1.0, units='m')   # initial valut of integrand (fltcond|h) = 0 by default.
        integ.add_integrand('mission_time', rate_name='dt_dt', val=0.0, units='s')

        # atmosphere model
        self.add_subsystem('atmos', ComputeAtmosphericProperties(num_nodes=nn, true_airspeed_in=False), promotes_inputs=['*'], promotes_outputs=['*'])

        # compute ground speed and flight path angle
        self.add_subsystem('gs', Groundspeeds(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['*'])
        integ.add_integrand('range', rate_name='fltcond|groundspeed', val=1.0, units='m')

        # angle of attack: alpha = -gamma + body_geom_alpha, where body_geom_alpha is the body AoA w.r.t. horizontal plane. Arcsin works if -90 <= gamma <= 90
        sincos_dict = {'shape' : (nn,)}
        self.add_subsystem('angle_of_attack', ExecComp('alpha = -arcsin(singamma) + body_geom_alpha', alpha={'units' : 'rad', 'shape' : (nn,)}, singamma=sincos_dict, body_geom_alpha={'units' : 'rad', 'shape' : (1,)}),
                            promotes_inputs=[('singamma', 'fltcond|singamma'), 'body_geom_alpha'], promotes_outputs=[('alpha', 'fltcond|alpha')])

        # add the user-defined aircraft model
        self.add_subsystem('acmodel', self.options['aircraft_model'](num_nodes=nn, flight_phase='transition'), promotes_inputs=['*'], promotes_outputs=['*'])

        # --- compute accelerations ---
        # thrust components in the flight path and perpendicular directions
        # - flight-path projection: T cos(Tangle - gamma); cos(Tangle - gamma) = cos(Tangle) * cos(gamma) + sin(Tangle) * sin(gamma)
        # - perpendicular to the flight path: T sin(Tangle - gamma); sin(Tangle - gamma) = sin(Tangle) * cos(gamma) - cos(Tangle) * sin(gamma)
        thrust_projection = ['thrust_fp = thrust * (cos(Tangle) * cosgamma + sin(Tangle) * singamma)']
        thrust_dict = {'units' : 'N', 'shape' : (nn,)}
        self.add_subsystem('thrust_proj', ExecComp(thrust_projection, thrust_fp=thrust_dict, thrust=thrust_dict, Tangle={'units' : 'rad', 'shape' : (nn,)}, singamma=sincos_dict, cosgamma=sincos_dict),
                            promotes_inputs=['thrust', 'Tangle', ('cosgamma', 'fltcond|cosgamma'), ('singamma', 'fltcond|singamma')])

        # vertical acceleration
        self.add_subsystem('vaccel', VerticalAccelerationTiltRotor(num_nodes=nn), promotes_inputs=['*'], promotes_outputs=['accel_vert'])

        # "horizontal" acceleration: acceleration in the flight path direction (i.e. airspeed direction). Not really horizontal when flight path angle is large.
        # Along the flight path, EoM is: m haccel = T cos(Tangle) - D - mg sin(gamma)
        self.add_subsystem('haccel', HorizontalAcceleration(num_nodes=nn), promotes_inputs=['weight', 'drag', 'lift', 'fltcond|singamma', 'braking'], promotes_outputs=['accel_horiz'])
        self.connect('thrust_proj.thrust_fp', 'haccel.thrust')

        # --- solve acceleration residuals ---
        # Residuals: computed acceleration = differential of the given speed history. Outputs (i.e., implicit variables determined by nonlinear solver) are the throttle and Tangle
        self.add_subsystem('unsteadyflt1', BalanceComp(name='throttle', val=np.ones((nn,)) * 0.5, lower=0.001, upper=1.2, units=None, normalize=False, eq_units='m/s**2', rhs_name='accel_horiz', lhs_name='accel_horiz_target', rhs_val=np.ones((nn,))),
                            promotes_inputs=['accel_horiz', 'accel_horiz_target'], promotes_outputs=['throttle'])
        self.add_subsystem('unsteadyflt2', BalanceComp(name='Tangle', val=np.ones((nn,)) * 90, lower=0., upper=180., units='deg', normalize=False, eq_units='m/s**2', rhs_name='accel_vert', lhs_name='accel_vert_target', rhs_val=np.ones((nn,))),
                            promotes_inputs=['accel_vert', 'accel_vert_target'], promotes_outputs=['Tangle'])

        """
        # Altenatively, we may want to use a single BalanceComp
        # set [throttle, Tangle] such that [accel_horiz = accel_horiz_target, accel_vert = accel_vert_target]
        # first, concatenate the lhs and rhs variables
        self.add_subsystem('balance_lhs', ConcatVectorComp(num_nodes=nn), promotes_inputs=[('vec1', 'accel_horiz'), ('vec2', 'accel_vert')], promotes_outputs=[('vec_concat', 'accels')])
        self.add_subsystem('balance_rhs', ConcatVectorComp(num_nodes=nn), promotes_inputs=[('vec1', 'accel_horiz_target'), ('vec2', 'accel_horiz_vert')], promotes_outputs=[('vec_concat', 'accels_target')])
        # balance comp
        val_init = np.concatenate((np.ones(nn)*0.5, np.ones(nn)*np.pi/2))
        lower = np.zeros(2 * nn)
        upper = np.concatenate((np.ones(nn)*1.5, np.ones(nn)*np.pi))
        self.add_subsystem('unsteadyflt', om.BalanceComp(name='balance_imp_vars', val=val_init, lower=lower, upper=upper, units=None, normalize=False, eq_units='m/s**2', rhs_name='accels', lhs_name='accels_target', rhs_val=val_init),
                            promotes_inputs=['accels', 'accels_target'])
        # split implicit variables (which is the output of BalanceComp) into [throttle, fltcond|CL]
        self.add_subsystem('balance_output', SplitVectorComp(num_nodes=2 * nn), promotes_outputs=[('vec1', 'throttle'), ('vec2', 'Tangle')])
        self.connect('unsteadyflt.balance_imp_vars', 'balance_output.vec_concat')
        # """

        # --- find the body geometric AoA ---
        """ TODO: Newton not working very well
        # impose the continuity of the rotor tilt angle between transition and cruise. Adds one residual and the body geometric AoA as an implicit variable.
        if flight_phase == 'transition_climb':
            anchor_index = -1   # Tangle_transition[-1] = Tangle_cruise
        elif flight_phase == 'transition_descent':
            anchor_index = 0   # Tangle_transition[0] = Tangle_cruise
        else:
            raise RuntimeError('Set flight_phase = transition_climb or transition_descent in the transition phase.')

        self.add_subsystem('Tangle_target', om.ExecComp('Tangle_anchor = Tangle', Tangle_anchor={'units' : 'rad', 'shape' : (1,)}, Tangle={'units' : 'rad', 'shape' : (1,), 'src_indices' : anchor_index}), promotes_inputs=['Tangle'])
        self.add_subsystem('tilt_angle_continuity', om.BalanceComp(name='body_geom_alpha', val=11., lower=-15., upper=15., units='deg', normalize=False, eq_units='rad', rhs_name='Tangle_anchor', lhs_name='Tangle_cruise', rhs_val=10.),
                            promotes_inputs=['Tangle_cruise'], promotes_outputs=['body_geom_alpha'])
        self.connect('Tangle_target.Tangle_anchor', 'tilt_angle_continuity.Tangle_anchor')
        """
        
        # or, set the body geometric AoA manually
        ivcomp.add_output('body_geom_alpha', 5, units='deg')
        ivcomp.add_output('Tangle_cruise', 45, units='deg')