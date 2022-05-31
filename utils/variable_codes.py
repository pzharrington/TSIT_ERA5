# -*- coding: utf-8 -*-

vars_levels = [
    'geopotential', 'temperature', 
    'u_component_of_wind', 'v_component_of_wind',
    'specific_humidity'] 

vars_at_level = [
    '2m_temperature', 'total_precipitation', 'toa_incident_solar_radiation'
]


# rasp's var_dict
var_dict_rasp = {'geopotential': ('z', [50, 250, 500, 600, 700, 850, 925]), 'temperature': ('t', [50, 250, 500, 600, 700, 850, 925]),
'u_component_of_wind': ('u', [50, 250, 500, 600, 700, 850, 925]), 'v_component_of_wind': ('v', [50, 250, 500, 600, 700, 850, 925]),
'specific_humidity': ('q', [50, 250, 500, 600, 700, 850, 925]), 'toa_incident_solar_radiation': ('tisr', None), '2m_temperature': ('t2m', None),
'total_precipitation': ('tp', None), 'constants': ['lsm','orography','lat2d']}
train_years = ('1979', '2015')
valid_years = ('2016', '2016')
test_years = ('2017', '2018')
var_dict_io = {'geopotential': ('z', [500]), '2m_temperature': ('t2m', None), 'temperature': ('t', [850])}
output_vars = ['z_500', 't_850', 't2m']
output_vars_tp = ['tp']

vars2d = [
    '2m_temperature',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    'total_cloud_cover', 'total_precipitation',
    'toa_incident_solar_radiation',
    'temperature_850',
]

vars3d = [
    'geopotential', 'temperature',
    'specific_humidity', 'relative_humidity',
    'u_component_of_wind', 'v_component_of_wind',
    'vorticity', 'potential_vorticity',
]

codes = {
    'geopotential': 'z',
    'geopotential_500': 'z',
    'temperature': 't',
    'temperature_850': 't',
    'specific_humidity': 'q',
    'relative_humidity': 'r',
    'u_component_of_wind': 'u',
    'v_component_of_wind': 'v',
    'vorticity': 'vo',
    'potential_vorticity': 'pv',
    '2m_temperature': 't2m',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    'total_cloud_cover': 'tcc',
    'total_precipitation': 'tp',
    'toa_incident_solar_radiation': 'tisr',
}

code2var = {
    'z': 'geopotential',
    't': 'temperature',
    'q': 'specific_humidity',
    'r': 'relative_humidity',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'vo': 'vorticity',
    'pv': 'potential_vorticity',
    't2m': '2m_temperature',
    'u10': '10m_u_component_of_wind',
    'v10': '10m_v_component_of_wind',
    'tcc': 'total_cloud_cover',
    'tp': 'total_precipitation',
    'tisr': 'toa_incident_solar_radiation',
}

