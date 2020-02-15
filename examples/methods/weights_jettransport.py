from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent, IndepVarComp, Group
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp

"""
Weight estimation components/groups for jet airliners. 
Some default values are intended for narrow-body, but equations should be valid for wide-body as well. 

"""

# -------------------------------------
#  structural components
# -------------------------------------
class WingWeight_JetAirliner(ExplicitComponent):
    """
    Inputs:
    Outputs: W_wing
    Metadata: n_ult (ult load factor)
    """
    def initialize(self):
        # TODO: need to check the limited load factor...
        self.options.declare('n_ult', default=3.5*1.5, desc='Ultimate load factor (dimensionless)')

    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|weights|W_fuel_max', units='lb', desc='Fuel weight')
        self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='Reference wing area in sq ft')
        self.add_input('ac|geom|wing|AR', desc='Wing aspect ratio')
        self.add_input('ac|geom|wing|c4sweep', units='rad', desc='Quarter-chord sweep angle')
        self.add_input('ac|geom|wing|taper', desc='Wing taper ratio')
        self.add_input('ac|geom|wing|toverc', desc='Wing max thickness to chord ratio')
        self.add_output('W_wing', units='lb', desc='Wing weight')
        self.declare_partials(['W_wing'], ['*'])

    def compute(self, inputs, outputs):
        # source: Kroo AA241, Component Weights
        # NOTE: assuming S_ref = S_gross; c2sweep = c_ea_sweep = 0.87c4sweep (ea = elastic axis) 
        S_gross = inputs['ac|geom|wing|S_ref']
        ceasweep = 0.87 * inputs['ac|geom|wing|c4sweep']
        b = np.sqrt(S_gross * inputs['ac|geom|wing|AR'])  # span [ft]
        zfw = inputs['ac|weights|MTOW'] - inputs['ac|weights|W_fuel_max']  # zero fuel weight [lb]
        nume = self.options['n_ult'] * b**3 * np.sqrt(inputs['ac|weights|MTOW']*zfw) * (1+2*inputs['ac|geom|wing|taper'])
        deno = inputs['ac|geom|wing|toverc'] * S_gross * np.cos(ceasweep)**2 * (1+inputs['ac|geom|wing|taper'])
        W_wing = 4.22 * S_gross + 1.642e-6 * nume/deno
        outputs['W_wing'] = W_wing

    def compute_partials(self, inputs, J):
        S_gross = inputs['ac|geom|wing|S_ref']
        b = np.sqrt(S_gross * inputs['ac|geom|wing|AR'])  # span [ft]
        zfw = inputs['ac|weights|MTOW'] - inputs['ac|weights|W_fuel_max']  # zero fuel weight [lb]
        taper = inputs['ac|geom|wing|taper']
        nume = self.options['n_ult'] * b**3 * np.sqrt(inputs['ac|weights|MTOW']*zfw) * (1+2*taper)
        ceasweep = 0.87 * inputs['ac|geom|wing|c4sweep']
        deno = inputs['ac|geom|wing|toverc'] * S_gross * np.cos(ceasweep)**2 * (1+taper)

        J['W_wing', 'ac|weights|MTOW'] = 1.642e-6 * nume/deno / np.sqrt(inputs['ac|weights|MTOW']*zfw) * (0.5/np.sqrt(inputs['ac|weights|MTOW']*zfw) * (2*inputs['ac|weights|MTOW'] - inputs['ac|weights|W_fuel_max']))
        J['W_wing', 'ac|weights|W_fuel_max'] = (0.5 * 1.642e-6 * nume/deno / zfw) * -1. 
        J['W_wing', 'ac|geom|wing|S_ref'] = 4.22 - 1.642e-6 * nume/deno / S_gross + (1.642e-6 * nume/deno / b**3) * (3*b**2 * np.sqrt(inputs['ac|geom|wing|AR']) * 0.5*S_gross**-0.5)  
        J['W_wing', 'ac|geom|wing|AR'] = (1.642e-6 * nume/deno / b**3) * (3*b**2 * np.sqrt(S_gross) * 0.5*inputs['ac|geom|wing|AR']**-0.5)  
        J['W_wing', 'ac|geom|wing|c4sweep'] = (-1.642e-6 * nume/deno / np.cos(ceasweep)**2) * (-2*np.cos(ceasweep)*np.sin(ceasweep)) * 0.87
        J['W_wing', 'ac|geom|wing|taper'] = 1.642e-6 * nume/deno * ((1+taper) / (1+2*taper)) / (1+taper)**2
        J['W_wing', 'ac|geom|wing|toverc'] = -1.642e-6 * nume/deno / inputs['ac|geom|wing|toverc']

class HorizontalTailWeight_JetAirliner(ExplicitComponent):
    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|hstab|S_ref', units='ft**2', desc='Horizontal tail area')
        self.add_input('ac|geom|hstab|AR', desc='Aspect ratio')
        self.add_input('ac|geom|hstab|c4sweep', units='rad', desc='Mid-chord sweep angle') 
        self.add_input('ac|geom|hstab|taper', desc='Taper ratio')
        self.add_input('ac|geom|hstab|toverc', desc='Thickness to chord ratio')   # NOTE: average? max?
        self.add_output('W_hstab', units='lb', desc='Horizontal tail weight')
        self.declare_partials(['W_hstab'], ['*'])

    def compute(self, inputs, outputs):
        # source: pyACDT (original source unknown)
        # NOTE: assuming S_ref = S_gross; c2sweep = 0.81 * c4sweep (based on A320 rough calc)
        c2sweep = 0.81 * inputs['ac|geom|hstab|c4sweep']
        term1 = (inputs['ac|geom|hstab|AR'] / np.cos(c2sweep)**2)**0.539
        term2 = ((1+inputs['ac|geom|hstab|taper']) / inputs['ac|geom|hstab|toverc'])**0.692
        term3 = (1 + 0.25)**0.1  # assuming that elevetor area is 25% of horizontal tail area. 
        W_hstab = 0.00563 * inputs['ac|geom|hstab|S_ref']**0.469 * inputs['ac|weights|MTOW']**0.6 * term1*term2*term3
        outputs['W_hstab'] = W_hstab

    def compute_partials(self, inputs, J):
        Sgross = inputs['ac|geom|hstab|S_ref']
        MTOW = inputs['ac|weights|MTOW']
        c2sweep = 0.81 * inputs['ac|geom|hstab|c4sweep']
        taper = inputs['ac|geom|hstab|taper']
        toverc = inputs['ac|geom|hstab|toverc']
        AR = inputs['ac|geom|hstab|AR']
        term1 = (AR / np.cos(c2sweep)**2)**0.539
        term2 = ((1+taper) / toverc)**0.692
        term3 = (1 + 0.25)**0.1  # assuming that elevetor area is 25% of horizontal tail area. 

        J['W_hstab', 'ac|weights|MTOW'] = 0.00563 * Sgross**0.469 * term1*term2*term3 * (0.6 * MTOW**(0.6-1))
        J['W_hstab', 'ac|geom|hstab|S_ref'] = 0.00563 * MTOW**0.6 * term1*term2*term3 * (0.469 * Sgross**(0.469-1))
        J['W_hstab', 'ac|geom|hstab|AR'] = 0.00563 * Sgross**0.469 * MTOW**0.6 * term2*term3 * (0.539*(AR / np.cos(c2sweep)**2)**(0.539-1) / np.cos(c2sweep)**2)
        J['W_hstab', 'ac|geom|hstab|c4sweep'] =  0.00563 * Sgross**0.469 * MTOW**0.6 * term2*term3 * AR**0.539 * (-1.078*np.cos(c2sweep)**(-1.078-1) * -np.sin(c2sweep)) * 0.81
        J['W_hstab', 'ac|geom|hstab|taper'] =  0.00563 * Sgross**0.469 * MTOW**0.6 * term1*term3 * (0.692*((1+taper)/toverc)**(0.692-1) / toverc)
        J['W_hstab', 'ac|geom|hstab|toverc'] =  0.00563 * Sgross**0.469 * MTOW**0.6 * term1*term3 * (1+taper)**0.692 * (-0.692*toverc**(-0.692-1)) 

class VerticalTailWeight_JetAirliner(ExplicitComponent):
    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_input('ac|geom|vstab|S_ref', units='ft**2', desc='Vertical tail area')
        self.add_input('ac|geom|vstab|AR', desc='Aspect ratio')
        self.add_input('ac|geom|vstab|c4sweep', units='rad', desc='Mid-chord sweep angle')
        self.add_input('ac|geom|vstab|taper', desc='Taper ratio')
        self.add_input('ac|geom|vstab|toverc', desc='Thickness to chord ratio')   # NOTE: average? max?
        self.add_output('W_vstab', units='lb', desc='Vertical tail weight')
        self.declare_partials(['W_vstab'], ['*'])

    def compute(self, inputs, outputs):
        # source: pyACDT (original source unknown)
        # NOTE: assuming S_ref = S_gross; c2sweep = 0.81 * c4sweep (based on A320 rough calc)
        c2sweep = 0.81 * inputs['ac|geom|vstab|c4sweep']
        term1 = (inputs['ac|geom|vstab|AR'] / np.cos(c2sweep)**2)**0.35
        term2 = ((1+inputs['ac|geom|vstab|taper']) / inputs['ac|geom|vstab|toverc'])**0.5
        term3 = (1 + 0.25)**0.1  # assuming that elevetor area is 25% of horizontal tail area. 
        W_vstab = 0.0909 * inputs['ac|geom|vstab|S_ref']**0.7* inputs['ac|weights|MTOW']**0.333 * term1*term2*term3
        outputs['W_vstab'] = W_vstab

    def compute_partials(self, inputs, J):
        Sgross = inputs['ac|geom|vstab|S_ref']
        MTOW = inputs['ac|weights|MTOW']
        c2sweep = 0.81 * inputs['ac|geom|vstab|c4sweep']
        taper = inputs['ac|geom|vstab|taper']
        toverc = inputs['ac|geom|vstab|toverc']
        AR = inputs['ac|geom|vstab|AR']
        term1 = (AR / np.cos(c2sweep)**2)**0.35
        term2 = ((1+taper) / toverc)**0.5
        term3 = (1 + 0.25)**0.1  # assuming that elevetor area is 25% of horizontal tail area. 

        J['W_vstab', 'ac|weights|MTOW'] = 0.0909 * Sgross**0.7 * term1*term2*term3 * (0.333 * MTOW**(0.333-1))
        J['W_vstab', 'ac|geom|vstab|S_ref'] = 0.0909 * MTOW**0.333 * term1*term2*term3 * (0.7 * Sgross**(0.7-1))
        J['W_vstab', 'ac|geom|vstab|AR'] = 0.0909 * Sgross**0.7 * MTOW**0.333 * term2*term3 * (0.35*(AR / np.cos(c2sweep)**2)**(0.35-1) / np.cos(c2sweep)**2)
        J['W_vstab', 'ac|geom|vstab|c4sweep'] =  0.0909 * Sgross**0.7 * MTOW**0.333 * term2*term3 * AR**0.35 * (-0.7*np.cos(c2sweep)**(-0.7-1) * -np.sin(c2sweep)) * 0.81
        J['W_vstab', 'ac|geom|vstab|taper'] =  0.0909 * Sgross**0.7 * MTOW**0.333 * term1*term3 * (0.5*((1+taper)/toverc)**(0.5-1) / toverc)
        J['W_vstab', 'ac|geom|vstab|toverc'] =  0.0909 * Sgross**0.7 * MTOW**0.333 * term1*term3 * (1+taper)**0.5 * (-0.5*toverc**(-0.5-1)) 


class LandingGearWeight_JetAirliner(ExplicitComponent):
    def setup(self):
        self.add_input('ac|weights|MTOW', units='lb', desc='Maximum rated takeoff weight')
        self.add_output('W_gear', units='lb', desc='Gear weight (nose and main)')
        self.declare_partials(['W_gear'], ['*'])

    def compute(self, inputs, outputs):
        # source: modified based on Kroo AA241, Component Weights 
        outputs['W_gear'] = 0.04 * inputs['ac|weights|MTOW'] 

    def compute_partials(self, inputs, J):
        J['W_gear', 'ac|weights|MTOW'] = 0.04

class FuselageWeight_JetAirliner(ExplicitComponent):
    def initialize(self):
        self.options.declare('Kw', default=1., desc='weight correction factor')

    def setup(self):
        # source: pyACDT (original source unknown)
        self.add_input('ac|geom|fuselage|length', units='ft', desc='Fuselage length')
        self.add_input('ac|geom|fuselage|height', units='ft', desc='Fuselage height')
        self.add_input('ac|geom|fuselage|width', units='ft', desc='Fuselage width')
        self.add_output('W_fuselage', units='lb', desc='Fuselage weight')
        self.declare_partials(['W_fuselage'], ['*'])

    def compute(self, inputs, outputs):
        # source: pyACDT (original source unknown)
        W_fuselage = 1.35 * self.options['Kw'] * (inputs['ac|geom|fuselage|length'] * (inputs['ac|geom|fuselage|height']+inputs['ac|geom|fuselage|width'])/2)**1.28
        outputs['W_fuselage'] = W_fuselage

    def compute_partials(self, inputs, J):
        Kw = self.options['Kw']
        Lf = inputs['ac|geom|fuselage|length']
        Hf = inputs['ac|geom|fuselage|height']
        Wf = inputs['ac|geom|fuselage|width']
        tmp = 1.35 * Kw * 1.28*(Lf * (Hf + Wf)/2)**(1.28-1)

        J['W_fuselage', 'ac|geom|fuselage|length'] = tmp * (Hf + Wf)/2
        J['W_fuselage', 'ac|geom|fuselage|height'] = tmp * Lf/2
        J['W_fuselage', 'ac|geom|fuselage|width'] = tmp * Lf/2

class NacelleWeight_JetAirliner(ExplicitComponent):
    def setup(self):
        self.add_input('ac|propulsion|max_thrust', units='lbf', desc='Engine maximum thrust')
        self.add_input('ac|geom|nacelle|diameter', units='ft', desc='Nacelle diameter')
        self.add_input('ac|geom|nacelle|length', units='ft', desc='Nacelle length')
        self.add_input('ac|propulsion|num_engine', val=2, desc='Number of engines. Default: twin') 
        self.add_output('W_nacelle', units='lb', desc='Naccele weight')
        self.declare_partials(['W_nacelle'], ['ac|propulsion|max_thrust', 'ac|geom|nacelle|diameter', 'ac|geom|nacelle|length'])

    def compute(self, inputs, outputs):
        # source: pyACDT (original source unknown)
        W_nacelle = 0.25 * inputs['ac|geom|nacelle|diameter'] * inputs['ac|geom|nacelle|length'] * inputs['ac|propulsion|max_thrust']**0.36
        outputs['W_nacelle'] = W_nacelle * inputs['ac|propulsion|num_engine']

    def compute_partials(self, inputs, J):
        n_engine = inputs['ac|propulsion|num_engine']
        J['W_nacelle', 'ac|propulsion|max_thrust'] = n_engine * 0.25 * inputs['ac|geom|nacelle|diameter'] * inputs['ac|geom|nacelle|length'] * 0.36 * inputs['ac|propulsion|max_thrust']**(0.36-1)
        J['W_nacelle', 'ac|geom|nacelle|diameter'] = n_engine * 0.25 * inputs['ac|geom|nacelle|length'] * inputs['ac|propulsion|max_thrust']**0.36
        J['W_nacelle', 'ac|geom|nacelle|length'] = n_engine * 0.25 * inputs['ac|geom|nacelle|diameter'] * inputs['ac|propulsion|max_thrust']**0.36

# -------------------------------------
#  system, operation, and payload
# -------------------------------------

class EquipmentWeight_JetAirliner(ExplicitComponent):
    # Weight estimation of misc. systems and equipments. Hardcoded for mid-range (domestic) flight.
    # Source: Kroo AA241, Component Weights, 8 - 14
    
    def setup(self):
        self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='wing reference area')
        self.add_input('ac|misc|num_passengers_max', val=180, desc='max number of passengers')
        self.add_input('ac|misc|num_crew', val=2, desc='number of crews')
        self.add_input('ac|misc|num_attend', val=5, desc='nuber of attendants')
        self.add_output('W_equip', val=30000., units='lb')
        self.declare_partials('W_equip', ['*'])

    def compute(self, inputs, outputs):
        n_seats = inputs['ac|misc|num_passengers_max'] + inputs['ac|misc|num_crew'] + inputs['ac|misc|num_attend']
        n_pax = inputs['ac|misc|num_passengers_max']
        # 1. APU
        W_apu = 7 * n_seats
        # 2. Instruments and Navigation Equipment
        W_ins = 800   # 1200 for long range, 800 for domestic
        # 3. Hydraulics and pneumatics
        W_hyd = 0.65 * inputs['ac|geom|wing|S_ref']
        # 4. Electrical 
        W_electrical = 13 * n_seats
        # 5. Electronics
        W_electronics = 900  # 1500 for log range, 900 for domestic
        # 6. Furnishing
        W_fur = (43.7 - 0.037*n_seats) * n_seats + 46*n_seats
        # 7. air-conditioning and anti-ice
        W_ac = 15 * n_pax 

        outputs['W_equip'] = W_apu + W_ins + W_hyd + W_electrical + W_electronics + W_fur + W_ac 

    def compute_partials(self, inputs, J):
        n_seats = inputs['ac|misc|num_passengers_max'] + inputs['ac|misc|num_crew'] + inputs['ac|misc|num_attend']
        J['W_equip', 'ac|geom|wing|S_ref'] = 0.65
        J['W_equip', 'ac|misc|num_passengers_max'] = 7 + 13 + 46 + 43.7 - 0.037*2*n_seats + 15
        J['W_equip', 'ac|misc|num_crew'] = 7 + 13 + 46 + 43.7 - 0.037*2*n_seats
        J['W_equip', 'ac|misc|num_attend'] = 7 + 13 + 46 + 43.7 - 0.037*2*n_seats


class OperationalWeight_JetAirliner(ExplicitComponent):
    # Weight estimation of operational weights. Hardcoded for mid-range (domestic) flight.
    # Source: Kroo AA241, Component Weights, 15 - 17

    def setup(self):
        #self.add_input('ac|propulsion|max_thrust', units='lbf', desc='Engine maximum thrust')
        #self.add_input('ac|geom|wing|S_ref', units='ft**2', desc='wing reference area')
        #self.add_input('ac|weights|W_fuel_max', units='lb', desc='Fuel weight')
        # self.add_input('ac|geom|nacelle|diameter', units='ft', desc='Nacelle diameter')
        #self.add_input('ac|propulsion|num_engine', val=2, desc='Number of engines. Default: twin') 
        #self.add_input('ac|propulsion|num_tank', val=3, desc='Number of fuel tanks') 
        self.add_input('ac|misc|num_passengers_max', val=180)
        self.add_input('ac|misc|num_crew', val=2, desc='number of flight deck crews')
        self.add_input('ac|misc|num_attend', val=5, desc='number of cabin attendants')
        self.add_output('W_op', units='lb', desc='operation weights')
        self.declare_partials(['W_op'], ['*'])

    def compute(self, inputs, outputs):
        # 0. unusable fuel weight
        # hardcoded for N_wing = 2
        #n_engine = inputs['ac|propulsion|num_engine']
        #n_tank = inputs['ac|propulsion|num_tank']
        #W_uufuel = 11.5*n_engine*inputs['ac|propulsion|max_thrust']**0.2 + 0.07*2*inputs['ac|geom|wing|S_ref'] + 1.6*n_tank*inputs['ac|weights|W_fuel_max']**0.28
        
        # 1. Operating items (passenger provisions)
        W_prov = 28 * inputs['ac|misc|num_passengers_max']  # 28 lb/pax for mid-range flight
        # 2. Deck crews: (180+25) per each
        W_crew = 205 * inputs['ac|misc|num_crew'] 
        # 3. Cabin attendants: (130+20) per each
        W_attend = 150 * inputs['ac|misc|num_attend']

        outputs['W_op'] = W_prov + W_crew + W_attend

    def compute_partials(self, inputs, J):
        J['W_op', 'ac|misc|num_passengers_max'] = 28.
        J['W_op', 'ac|misc|num_crew'] = 205.
        J['W_op', 'ac|misc|num_attend'] = 150.

class PayloadWeight_JetAirliner(ExplicitComponent):
    # Weight estimation of payload, i.e. passengers and baggages
    # Source: Kroo AA241, Component Weights, 18
    
    def setup(self):
        self.add_input('ac|misc|num_passengers_max', val=180, desc='max number of passengers')
        self.add_output('W_payload', val=200*205., units='lb')
        self.declare_partials('W_payload', 'ac|misc|num_passengers_max')

    def compute(self, inputs, outputs):
        # 205 lb / passenger (165 per person & 40lbs baggage)
        outputs['W_payload'] = 205. * inputs['ac|misc|num_passengers_max']
    
    def compute_partials(self, inputs, J):
        J['W_payload', 'ac|misc|num_passengers_max'] = 205. 

# -------------------------------------
#  engine components
# -------------------------------------
class EngineWeight_TurbofanSystem(ExplicitComponent):
    def setup(self):
        self.add_input('ac|propulsion|max_thrust', units='lbf', desc='Maximum thrust per engine')
        self.add_input('ac|propulsion|num_engine', val=2, desc='Number of engines. Default: twin')  
        self.add_output('W_engines', units='lb', desc=' engines weight (n engines total)')
        self.declare_partials('W_engines', 'ac|propulsion|max_thrust')

    def compute(self, inputs, outputs):
        # source: Roskam
        n_engine = inputs['ac|propulsion|num_engine']
        T0 = inputs['ac|propulsion|max_thrust']
        W_dry = 0.521 * T0**0.9
        W_oil = 0.082 * T0**0.65
        W_rev = 0.034 * T0
        W_con = 0.260 * T0**0.5
        W_start = 9.33 * (W_dry/1000)**1.078
        outputs['W_engines'] = n_engine * (W_dry + W_oil + W_rev + W_con + W_start) 

    def compute_partials(self, inputs, J):
        n_engine = inputs['ac|propulsion|num_engine']
        T0 = inputs['ac|propulsion|max_thrust']
        W_dry = 0.521 * T0**0.9
        J_Wdry_T0 = 0.521*0.9*T0**(0.9-1)
        J['W_engines', 'ac|propulsion|max_thrust'] = n_engine * (J_Wdry_T0 + 0.082*0.65*T0**(0.65-1) + 0.034 + 0.260*0.5*T0**(0.5-1) + 9.33*1.078*(W_dry/1000)**(1.078-1)/1000 * J_Wdry_T0)

class JetAirlinerEmptyWeight(Group):
    # computes OEW: Operational empty weight
    # NOTE: works good for A320-200 with 180 pax, but may need to tune fudge factors for other aircrafts
    def setup(self):
        # set fudge factors on structure / equipment weight
        const = self.add_subsystem('const', IndepVarComp(), promotes_outputs=["*"])
        const.add_output('structural_fudge', val=1., units='m/m')
        const.add_output('equipment_fudge', val=1.0, units='m/m')
        # structural weight
        self.add_subsystem('wing', WingWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('hstab', HorizontalTailWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('vstab', VerticalTailWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('fuselage', FuselageWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('nacelle', NacelleWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('gear', LandingGearWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('structural', AddSubtractComp(output_name='W_structure',input_names=['W_wing','W_fuselage','W_nacelle','W_hstab','W_vstab','W_gear'], units='lb'),promotes_outputs=['*'],promotes_inputs=["*"])
        self.add_subsystem('structural_fudge',ElementMultiplyDivideComp(output_name='W_structure_adjusted',input_names=['W_structure','structural_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])
        # engine weight
        self.add_subsystem('engines', EngineWeight_TurbofanSystem(), promotes_inputs=["*"], promotes_outputs=["*"])
        # other weights
        self.add_subsystem('equipment', EquipmentWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('operation', OperationalWeight_JetAirliner(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem('equipment_fudge',ElementMultiplyDivideComp(output_name='W_equip_adjusted',input_names=['W_equip','equipment_fudge'],input_units=['lb','m/m']),promotes_inputs=["*"],promotes_outputs=["*"])    
        # sum
        self.add_subsystem('totalempty', AddSubtractComp(output_name='OEW',input_names=['W_structure_adjusted', 'W_engines', 'W_equip_adjusted', 'W_op'], units='lb'), promotes_outputs=['*'], promotes_inputs=["*"])
     

