import openconcept.api as oc
from openconcept.analysis.performance.solver_phases_vtol import SteadyVerticalFlightPhase, SteadyFlightPhaseForVTOLCruise, UnsteadyFlightPhaseForTiltrotorTransition

class SimpleVTOLMission(oc.TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, hover, and vertical descent.
    The user needs to set the duration and vertical speed (fltcond|vs) of each phase in the runscript.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        hover = self.add_subsystem('hover', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='hover') , promotes_inputs=['ac|*'])
        descent1 = self.add_subsystem('descent1', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
        descent2 = self.add_subsystem('descent2', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(climb, hover)
        self.link_phases(hover, descent1)
        self.link_phases(descent1, descent2)


class SimpleVTOLMissionWithCruise(oc.TrajectoryGroup):
    """
    Simple VTOL mission, including vertical climb, cruise1, hover, cruise2, and vertical descent.
    The user can to set the followings in runscript
        - in climb/hover/descent, [duration, fltcond|vs]
        - in cruise, [duration, fltcond|vs, fltcond|Ueas, Tangle]
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']

        # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
        climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
        cruise1 = self.add_subsystem('cruise1', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        hover = self.add_subsystem('hover', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='hover') , promotes_inputs=['ac|*'])
        cruise2 = self.add_subsystem('cruise2', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
        descent1 = self.add_subsystem('descent1', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
        descent2 = self.add_subsystem('descent2', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

        # connect bettery SOC, altitude, and mission_time of each segments
        self.link_phases(climb, cruise1)
        self.link_phases(cruise1, hover)
        self.link_phases(hover, cruise2)
        self.link_phases(cruise2, descent1)
        self.link_phases(descent1, descent2)


class SimpleVTOLMissionWithTransition(oc.TrajectoryGroup):
    """
    VTOL mission, including vertical climb, transition1, cruise, transition2, and vertical descent.
    The user can to set the followings in runscript
        - in climb/hover/descent, [duration, fltcond|vs]
        - in cruise, [duration, fltcond|vs, fltcond|Ueas, Tangle]
        - in transition, [duration, fltcond|vs, fltcond|Ueas, accel_horiz_target, accel_vert_target, Tangle_cruise, body_geom_alpha]
    """

    def initialize(self):
        self.options.declare('num_nodes', default=9, desc="Number of points per segment. Needs to be 2N + 1 due to simpson's rule")
        self.options.declare('aircraft_model', default=None, desc="OpenConcept-compliant airplane model")
        self.options.declare('mode', default='full', desc="full or takeoff or landing")
        # full: vertical climb, transition1, cruise, transition2, and vertical descent.
        # takeoff: exclude transition 2
        # landing: exclude transition 1

    def setup(self):
        nn = self.options['num_nodes']
        acmodelclass = self.options['aircraft_model']
        mode = self.options['mode']

        if mode == 'full':
            # add climb, hover, and descent segments. Promote ac|* (such as W_battery, motor rating, prop diameter)
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            tran1 = self.add_subsystem('transition1', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_climb'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            tran2 = self.add_subsystem('transition2', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_descent'), promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])

            # connect bettery SOC, altitude, and mission_time of each segments
            self.link_phases(climb, tran1)
            self.link_phases(tran1, cruise)
            self.link_phases(cruise, tran2)
            self.link_phases(tran2, descent)

        elif mode == 'takeoff':
            # transition in takeoff only
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            tran1 = self.add_subsystem('transition1', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_climb'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
            self.link_phases(climb, tran1)
            self.link_phases(tran1, cruise)
            self.link_phases(cruise, descent)

        elif mode == 'landing':
            climb = self.add_subsystem('climb', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='climb'), promotes_inputs=['ac|*'])
            cruise = self.add_subsystem('cruise', SteadyFlightPhaseForVTOLCruise(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='cruise') , promotes_inputs=['ac|*'])
            tran2 = self.add_subsystem('transition2', UnsteadyFlightPhaseForTiltrotorTransition(num_nodes=nn * 3, aircraft_model=acmodelclass, flight_phase='transition_descent'), promotes_inputs=['ac|*'])
            descent = self.add_subsystem('descent', SteadyVerticalFlightPhase(num_nodes=nn, aircraft_model=acmodelclass, flight_phase='descent'), promotes_inputs=['ac|*'])
            self.link_phases(climb, cruise)
            self.link_phases(cruise, tran2)
            self.link_phases(tran2, descent)