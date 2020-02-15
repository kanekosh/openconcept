# DATA FOR Airbus 320-200
# Source: Civil Jet Aircraft Design, Data A, Table 1, otherwise stated.

from __future__ import division

data = dict()
ac = dict()
# ==AERO==================================
# TODO: modifiy AERO data!
aero = dict()
aero['CLmax_TO']   = {'value' : 1.7}

polar = dict()
polar['e']              = {'value' : 0.8}  # guesstimate
polar['CD0_TO']         = {'value' : 0.0276}
polar['CD0_cruise']     = {'value' : 0.0176}  # guestimate from pyACDT

aero['polar'] = polar
ac['aero'] = aero

# ==GEOMETRY==============================
geom = dict()
wing = dict()
# NOTE: want wing gross area for weight estimation. but for aero...?
wing['S_ref']           = {'value': 1530, 'units': 'ft**2'}  # reference-area, SC: pyACDT geom. (reference area)
# b = 34.10 * 3.28084 # span [ft], source: Airbus AC-A320
# aspect_ratio = b**2 / wing['S_ref']  ['value']
wing['AR']              = {'value': 8.1}  
wing['c4sweep']         = {'value': 25., 'units': 'deg'} 
wing['taper']           = {'value': 0.24}
wing['toverc']          = {'value': 0.106}  # SC: pyACDT 
geom['wing'] = wing

hstab = dict()
hstab['S_ref']          = {'value': 31, 'units': 'm**2'}
hstab['AR']             = {'value': 5.0}
hstab['c4sweep']        = {'value': 29., 'units': 'deg'}  
hstab['taper']          = {'value': 0.26}
hstab['toverc']         = {'value': 0.09}  # SC: pyACDT
# hstab['c4_to_wing_c4']  = {'value': 17.9, 'units': 'ft'}
geom['hstab'] = hstab

vstab = dict()
vstab['S_ref']          = {'value': 21.5, 'units': 'm**2'}
vstab['AR']             = {'value': 1.82}
vstab['c4sweep']        = {'value': 34., 'units': 'deg'}  
vstab['taper']          = {'value': 0.303}
vstab['toverc']         = {'value': 0.09}  # SC: pyACDT
geom['vstab'] = vstab

fuselage = dict()
fuselage['length']      = {'value': 37.57, 'units': 'm'} 
fuselage['width']       = {'value': 3.95, 'units': 'm'}
fuselage['height']      = {'value': 4.14, 'units': 'm'}
geom['fuselage'] = fuselage

nacelle = dict()
nacelle['diameter']     = {'value': 2.37, 'units': 'm'}
nacelle['length']       = {'value': 4.44, 'units': 'm'}
geom['nacelle'] = nacelle

#nosegear = dict()
# nosegear['length'] = {'value': 3, 'units': 'ft'}
# geom['nosegear'] = nosegear

# maingear = dict()
# maingear['length'] = {'value': 4, 'units': 'ft'}
# geom['maingear'] = maingear

ac['geom'] = geom

# ==WEIGHTS========================
weights = dict()
weights['MTOW']         = {'value': 172000, 'units': 'lb'}  # SC: wikipedia <- Airbus
weights['W_fuel_max']   = {'value': 46308, 'units': 'lb'}   # SC: ?
weights['MLW']          = {'value': 127000, 'units': 'lb'}  # SC: modernairliners.com

ac['weights'] = weights

# ==PROPULSION=====================
propulsion = dict()
propulsion['max_thrust']        = {'value': 25000, 'units': 'lbf'}  # per engine
propulsion['num_engine']        = {'value': 2}  # per engine

ac['propulsion'] = propulsion

# Some additional parameters needed by the empirical weights tools
misc = dict()
misc['num_passengers_max'] = {'value': 180}


ac['misc'] = misc
data['ac'] = ac