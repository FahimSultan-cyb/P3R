import numpy as np
import pandas as pd
import json
import os
from datetime import datetime

class SpaceMissionEvaluator:
    def __init__(self):
        self.orbit_types = ['LEO', 'MEO', 'GEO', 'HEO', 'SSO']
        self.spacecraft_properties = {
            'mass': (100, 10000),
            'cross_section': (1, 100),
            'reflectivity': (0.1, 0.9),
            'radar_cross_section': (0.1, 10.0)
        }
    
    def generate_mission_profile(self, num_samples):
        profiles = []
        for _ in range(num_samples):
            orbit_type = np.random.choice(self.orbit_types)
            altitude = self._get_altitude(orbit_type)
            inclination = self._get_inclination(orbit_type)
            
            profile = {
                'orbit_type': orbit_type,
                'altitude_km': altitude,
                'inclination_deg': inclination,
                'mass_kg': np.random.uniform(*self.spacecraft_properties['mass']),
                'cross_section_m2': np.random.uniform(*self.spacecraft_properties['cross_section']),
                'reflectivity': np.random.uniform(*self.spacecraft_properties['reflectivity']),
                'radar_cross_section': np.random.uniform(*self.spacecraft_properties['radar_cross_section']),
                'eccentricity': np.random.uniform(0, 0.2)
            }
            profiles.append(profile)
        return profiles
    
    def _get_altitude(self, orbit_type):
        altitude_ranges = {
            'LEO': (160, 2000),
            'MEO': (2000, 35786),
            'GEO': (35786, 35786),
            'HEO': (1000, 40000),
            'SSO': (600, 800)
        }
        low, high = altitude_ranges.get(orbit_type, (400, 400))
        return np.random.uniform(low, high) if low != high else low
    
    def _get_inclination(self, orbit_type):
        inclination_ranges = {
            'LEO': (0, 90),
            'MEO': (45, 55),
            'GEO': (0, 0),
            'HEO': (30, 60),
            'SSO': (98, 98)
        }
        low, high = inclination_ranges.get(orbit_type, (0, 0))
        return np.random.uniform(low, high) if low != high else low
    
    def calculate_dit_scores(self, mission_profiles):
        dit_scores = []
        for profile in mission_profiles:
            detectability = self._calculate_detectability(profile)
            identifiability = self._calculate_identifiability(profile)
            trackability = self._calculate_trackability(profile)
            
            dit_score = {
                'detectability': detectability,
                'identifiability': identifiability,
                'trackability': trackability,
                'overall_score': (detectability + identifiability + trackability) / 3
            }
            dit_scores.append(dit_score)
        return dit_scores
    
    def _calculate_detectability(self, profile):
        visual_mag = 20 - 2.5 * np.log10(profile['cross_section_m2'] * profile['reflectivity'] / (profile['altitude_km']**2))
        radar_prob = np.tanh(profile['radar_cross_section'] / profile['altitude_km'] * 100)
        return max(0, min(1, (visual_mag / 20) * 0.5 + radar_prob * 0.5))
    
    def _calculate_identifiability(self, profile):
        return 1.0 / (1 + np.exp(-0.1 * (profile['mass_kg'] / 1000 + profile['cross_section_m2'])))
    
    def _calculate_trackability(self, profile):
        access_duration = np.log(profile['altitude_km'] + 1) / 10
        access_frequency = 1 / (1 + np.exp(-0.001 * (2000 - profile['altitude_km'])))
        return max(0, min(1, access_duration * 0.6 + access_frequency * 0.4))

class KSPMissionSimulator:
    def __init__(self):
        self.mission_results = []
        
    def run_simulation(self, mission_duration=600):
        self.mission_results = []
        for t in range(mission_duration):
            mission_state = self._simulate_step(t)
            self.mission_results.append(mission_state)
        return self.mission_results
    
    def _simulate_step(self, time_step):
        return {
            'time': time_step,
            'altitude': np.random.uniform(100000, 400000),
            'velocity_magnitude': np.random.uniform(2000, 8000),
            'orbital_elements': {
                'apoapsis': np.random.uniform(70000, 90000),
                'periapsis': np.random.uniform(70000, 90000),
                'inclination': np.random.uniform(0, 90),
                'eccentricity': np.random.uniform(0, 0.2)
            },
            'thermal_properties': {
                'temperature': np.random.uniform(200, 1000),
                'radiative_cooling': np.random.uniform(0.1, 0.9),
                'heat_shield_integrity': np.random.uniform(0.5, 1.0)
            },
            'resource_management': {
                'fuel_percentage': max(0, 100 - (time_step / 6)),
                'delta_v_remaining': max(0, 10000 - (time_step * 10)),
                'mass_ratio': np.random.uniform(0.1, 0.9),
                'power_level': np.random.uniform(50, 100)
            },
            'science_experiments': {
                'science_multiplier': np.random.uniform(0.5, 2.0),
                'data_storage_used': min(100, time_step * 0.5)
            }
        }
    
    def analyze_impact(self):
        if not self.mission_results:
            return {}
        
        final_state = self.mission_results[-1]
        
        orbital_eff = 1 - abs(final_state['orbital_elements']['apoapsis'] - 80000) / 80000
        resource_eff = final_state['resource_management']['fuel_percentage'] / 100
        thermal_stable = final_state['thermal_properties']['temperature'] < 800
        
        return {
            'orbital_efficiency': max(0, orbital_eff),
            'fuel_efficiency': resource_eff,
            'thermal_stability': thermal_stable,
            'science_collection': len(final_state.get('science_experiments', {}))
        }
    
    def export_data(self, output_dir="data/outputs"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.mission_results:
            df = pd.DataFrame(self.mission_results)
            filepath = f"{output_dir}/ksp_mission_data_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            return filepath
        return None