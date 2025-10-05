# import numpy as np
# import pandas as pd
# import json
# import os
# from datetime import datetime

# class SpaceMissionEvaluator:
#     def __init__(self):
#         self.orbit_types = ['LEO', 'MEO', 'GEO', 'HEO', 'SSO']
#         self.spacecraft_properties = {
#             'mass': (100, 10000),
#             'cross_section': (1, 100),
#             'reflectivity': (0.1, 0.9),
#             'radar_cross_section': (0.1, 10.0)
#         }
    
#     def generate_mission_profile(self, num_samples):
#         profiles = []
#         for _ in range(num_samples):
#             orbit_type = np.random.choice(self.orbit_types)
#             altitude = self._get_altitude(orbit_type)
#             inclination = self._get_inclination(orbit_type)
            
#             profile = {
#                 'orbit_type': orbit_type,
#                 'altitude_km': altitude,
#                 'inclination_deg': inclination,
#                 'mass_kg': np.random.uniform(*self.spacecraft_properties['mass']),
#                 'cross_section_m2': np.random.uniform(*self.spacecraft_properties['cross_section']),
#                 'reflectivity': np.random.uniform(*self.spacecraft_properties['reflectivity']),
#                 'radar_cross_section': np.random.uniform(*self.spacecraft_properties['radar_cross_section']),
#                 'eccentricity': np.random.uniform(0, 0.2)
#             }
#             profiles.append(profile)
#         return profiles
    
#     def _get_altitude(self, orbit_type):
#         altitude_ranges = {
#             'LEO': (160, 2000),
#             'MEO': (2000, 35786),
#             'GEO': (35786, 35786),
#             'HEO': (1000, 40000),
#             'SSO': (600, 800)
#         }
#         low, high = altitude_ranges.get(orbit_type, (400, 400))
#         return np.random.uniform(low, high) if low != high else low
    
#     def _get_inclination(self, orbit_type):
#         inclination_ranges = {
#             'LEO': (0, 90),
#             'MEO': (45, 55),
#             'GEO': (0, 0),
#             'HEO': (30, 60),
#             'SSO': (98, 98)
#         }
#         low, high = inclination_ranges.get(orbit_type, (0, 0))
#         return np.random.uniform(low, high) if low != high else low
    
#     def calculate_dit_scores(self, mission_profiles):
#         dit_scores = []
#         for profile in mission_profiles:
#             detectability = self._calculate_detectability(profile)
#             identifiability = self._calculate_identifiability(profile)
#             trackability = self._calculate_trackability(profile)
            
#             dit_score = {
#                 'detectability': detectability,
#                 'identifiability': identifiability,
#                 'trackability': trackability,
#                 'overall_score': (detectability + identifiability + trackability) / 3
#             }
#             dit_scores.append(dit_score)
#         return dit_scores
    
#     def _calculate_detectability(self, profile):
#         visual_mag = 20 - 2.5 * np.log10(profile['cross_section_m2'] * profile['reflectivity'] / (profile['altitude_km']**2))
#         radar_prob = np.tanh(profile['radar_cross_section'] / profile['altitude_km'] * 100)
#         return max(0, min(1, (visual_mag / 20) * 0.5 + radar_prob * 0.5))
    
#     def _calculate_identifiability(self, profile):
#         return 1.0 / (1 + np.exp(-0.1 * (profile['mass_kg'] / 1000 + profile['cross_section_m2'])))
    
#     def _calculate_trackability(self, profile):
#         access_duration = np.log(profile['altitude_km'] + 1) / 10
#         access_frequency = 1 / (1 + np.exp(-0.001 * (2000 - profile['altitude_km'])))
#         return max(0, min(1, access_duration * 0.6 + access_frequency * 0.4))

# class KSPMissionSimulator:
#     def __init__(self):
#         self.mission_results = []
        
#     def run_simulation(self, mission_duration=600):
#         self.mission_results = []
#         for t in range(mission_duration):
#             mission_state = self._simulate_step(t)
#             self.mission_results.append(mission_state)
#         return self.mission_results
    
#     def _simulate_step(self, time_step):
#         return {
#             'time': time_step,
#             'altitude': np.random.uniform(100000, 400000),
#             'velocity_magnitude': np.random.uniform(2000, 8000),
#             'orbital_elements': {
#                 'apoapsis': np.random.uniform(70000, 90000),
#                 'periapsis': np.random.uniform(70000, 90000),
#                 'inclination': np.random.uniform(0, 90),
#                 'eccentricity': np.random.uniform(0, 0.2)
#             },
#             'thermal_properties': {
#                 'temperature': np.random.uniform(200, 1000),
#                 'radiative_cooling': np.random.uniform(0.1, 0.9),
#                 'heat_shield_integrity': np.random.uniform(0.5, 1.0)
#             },
#             'resource_management': {
#                 'fuel_percentage': max(0, 100 - (time_step / 6)),
#                 'delta_v_remaining': max(0, 10000 - (time_step * 10)),
#                 'mass_ratio': np.random.uniform(0.1, 0.9),
#                 'power_level': np.random.uniform(50, 100)
#             },
#             'science_experiments': {
#                 'science_multiplier': np.random.uniform(0.5, 2.0),
#                 'data_storage_used': min(100, time_step * 0.5)
#             }
#         }
    
#     def analyze_impact(self):
#         if not self.mission_results:
#             return {}
        
#         final_state = self.mission_results[-1]
        
#         orbital_eff = 1 - abs(final_state['orbital_elements']['apoapsis'] - 80000) / 80000
#         resource_eff = final_state['resource_management']['fuel_percentage'] / 100
#         thermal_stable = final_state['thermal_properties']['temperature'] < 800
        
#         return {
#             'orbital_efficiency': max(0, orbital_eff),
#             'fuel_efficiency': resource_eff,
#             'thermal_stability': thermal_stable,
#             'science_collection': len(final_state.get('science_experiments', {}))
#         }
    
#     def export_data(self, output_dir="data/outputs"):
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         if self.mission_results:
#             df = pd.DataFrame(self.mission_results)
#             filepath = f"{output_dir}/ksp_mission_data_{timestamp}.csv"
#             df.to_csv(filepath, index=False)
#             return filepath

#         return None




import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

class NASAMetricsCalculator:
    def __init__(self):
        self.criticality_weights = {
            'CATASTROPHIC': 1.0,
            'CRITICAL': 0.8,
            'MAJOR': 0.6,
            'MINOR': 0.4,
            'NEGLIGIBLE': 0.2
        }
        
        self.software_classes = {
            'A': {'mission_impact': 1.0, 'human_safety': True},
            'B': {'mission_impact': 0.85, 'human_safety': True},
            'C': {'mission_impact': 0.7, 'human_safety': False},
            'D': {'mission_impact': 0.5, 'human_safety': False},
            'E': {'mission_impact': 0.3, 'human_safety': False}
        }

    def calculate_detection_metrics(self, y_true, y_pred, y_proba):
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(y_true, y_proba)
        except:
            auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc': float(auc)
        }

    def calculate_vulnerability_severity(self, predictions, probabilities):
        critical_vulns = sum(1 for p, prob in zip(predictions, probabilities) 
                           if p == 1 and prob > 0.7)
        high_severity = sum(1 for p, prob in zip(predictions, probabilities) 
                          if p == 1 and 0.5 < prob <= 0.7)
        medium_severity = sum(1 for p, prob in zip(predictions, probabilities) 
                            if p == 1 and 0.3 < prob <= 0.5)
        low_severity = sum(1 for p, prob in zip(predictions, probabilities) 
                         if p == 1 and prob <= 0.3)
        
        return {
            'critical': int(critical_vulns),
            'high': int(high_severity),
            'medium': int(medium_severity),
            'low': int(low_severity),
            'total': int(sum(predictions))
        }

    def calculate_mission_risk_score(self, vulnerability_counts, total_samples):
        risk_score = (
            vulnerability_counts['critical'] * self.criticality_weights['CATASTROPHIC'] +
            vulnerability_counts['high'] * self.criticality_weights['CRITICAL'] +
            vulnerability_counts['medium'] * self.criticality_weights['MAJOR'] +
            vulnerability_counts['low'] * self.criticality_weights['MINOR']
        ) / max(total_samples, 1)
        
        return min(1.0, risk_score)

    def assess_software_assurance_level(self, risk_score, accuracy):
        if risk_score > 0.7 or accuracy < 0.80:
            return 'REQUIRES_REMEDIATION'
        elif risk_score > 0.5 or accuracy < 0.85:
            return 'MARGINAL'
        elif risk_score > 0.3 or accuracy < 0.92:
            return 'ACCEPTABLE'
        else:
            return 'EXCELLENT'

    def calculate_mission_readiness(self, risk_score, detection_metrics):
        risk_factor = 1.0 - risk_score
        accuracy_factor = detection_metrics['accuracy']
        precision_factor = detection_metrics['precision']
        recall_factor = detection_metrics['recall']
        
        readiness = (
            risk_factor * 0.35 +
            accuracy_factor * 0.25 +
            precision_factor * 0.20 +
            recall_factor * 0.20
        )
        
        return max(0.0, min(1.0, readiness))

    def compute_comprehensive_metrics(self, y_true, y_pred, y_proba):
        detection_metrics = self.calculate_detection_metrics(y_true, y_pred, y_proba)
        
        vulnerability_counts = self.calculate_vulnerability_severity(y_pred, y_proba)
        
        mission_risk = self.calculate_mission_risk_score(
            vulnerability_counts, len(y_true)
        )
        
        assurance_level = self.assess_software_assurance_level(
            mission_risk, detection_metrics['accuracy']
        )
        
        mission_readiness = self.calculate_mission_readiness(
            mission_risk, detection_metrics
        )
        
        return {
            'detection_metrics': detection_metrics,
            'vulnerability_counts': vulnerability_counts,
            'mission_risk_score': float(mission_risk),
            'software_assurance_level': assurance_level,
            'mission_readiness_score': float(mission_readiness)
        }

class SpacecraftSimulator:
    def __init__(self):
        self.subsystems = {
            'ADCS': {'criticality': 'CRITICAL', 'redundancy': 2},
            'CDH': {'criticality': 'CATASTROPHIC', 'redundancy': 3},
            'EPS': {'criticality': 'CATASTROPHIC', 'redundancy': 2},
            'PROP': {'criticality': 'CRITICAL', 'redundancy': 1},
            'COMM': {'criticality': 'MAJOR', 'redundancy': 2},
            'PAYLOAD': {'criticality': 'MAJOR', 'redundancy': 1}
        }
        
        self.nominal_params = {
            'altitude_km': 400.0,
            'velocity_mps': 7670.0,
            'power_w': 500.0,
            'temperature_k': 293.0
        }

    def calculate_fault_impact(self, subsystem, vulnerability_score):
        criticality_multiplier = {
            'CATASTROPHIC': 1.0,
            'CRITICAL': 0.8,
            'MAJOR': 0.6,
            'MINOR': 0.4,
            'NEGLIGIBLE': 0.2
        }
        
        subsystem_info = self.subsystems.get(subsystem, {'criticality': 'MAJOR', 'redundancy': 1})
        crit_mult = criticality_multiplier[subsystem_info['criticality']]
        redundancy_factor = 1.0 / subsystem_info['redundancy']
        
        return vulnerability_score * crit_mult * redundancy_factor

    def simulate_orbital_state(self, fault_impacts, mission_time):
        adcs_impact = fault_impacts.get('ADCS', 0)
        prop_impact = fault_impacts.get('PROP', 0)
        
        altitude = self.nominal_params['altitude_km'] * (1 - adcs_impact * 0.1 - prop_impact * 0.15)
        velocity = self.nominal_params['velocity_mps'] * (1 - prop_impact * 0.05)
        decay_rate = adcs_impact * 2.0 + prop_impact * 3.0
        
        time_factor = np.log1p(mission_time) / 100
        altitude *= (1 - time_factor)
        
        return {
            'altitude_km': max(100, altitude),
            'velocity_mps': velocity,
            'decay_rate_km_per_day': decay_rate
        }

    def simulate_power_state(self, fault_impacts, mission_time):
        eps_impact = fault_impacts.get('EPS', 0)
        cdh_impact = fault_impacts.get('CDH', 0)
        
        efficiency = 1.0 - (eps_impact * 0.3 + cdh_impact * 0.2)
        age_factor = mission_time * 0.0001
        power = self.nominal_params['power_w'] * efficiency * (1 - age_factor)
        
        return {
            'power_w': max(50, power),
            'efficiency': efficiency,
            'battery_health': 1.0 - eps_impact * 0.4
        }

    def simulate_thermal_state(self, fault_impacts, altitude):
        eps_impact = fault_impacts.get('EPS', 0)
        payload_impact = fault_impacts.get('PAYLOAD', 0)
        
        altitude_factor = 1.0 - (altitude / 50000)
        thermal_stress = eps_impact * 50 + payload_impact * 30
        temperature = self.nominal_params['temperature_k'] + thermal_stress * altitude_factor
        
        return {
            'temperature_k': temperature,
            'thermal_margin_k': max(0, 323 - temperature),
            'control_efficiency': 1.0 - eps_impact * 0.3
        }

    def simulate_communication_state(self, fault_impacts, altitude):
        comm_impact = fault_impacts.get('COMM', 0)
        cdh_impact = fault_impacts.get('CDH', 0)
        
        signal_base = -80 + (altitude / 1000) * -0.5
        signal_degradation = comm_impact * 15 + cdh_impact * 10
        signal_strength = signal_base - signal_degradation
        
        data_rate = 10 * (1 - comm_impact * 0.6)
        
        return {
            'signal_strength_db': signal_strength,
            'data_rate_mbps': max(0.1, data_rate),
            'link_availability': 1.0 - comm_impact * 0.5
        }

    def run_mission_simulation(self, predictions, probabilities, labels, duration_hours=720):
        mission_states = []
        subsystem_keys = list(self.subsystems.keys())
        
        samples_per_hour = max(1, len(predictions) // duration_hours)
        
        for hour in range(min(duration_hours, len(predictions))):
            idx = min(hour * samples_per_hour, len(predictions) - 1)
            
            fault_impacts = {}
            if predictions[idx] == 1:
                affected = subsystem_keys[hour % len(subsystem_keys)]
                impact = self.calculate_fault_impact(affected, probabilities[idx])
                fault_impacts[affected] = impact
            
            orbital = self.simulate_orbital_state(fault_impacts, hour)
            power = self.simulate_power_state(fault_impacts, hour)
            thermal = self.simulate_thermal_state(fault_impacts, orbital['altitude_km'])
            comm = self.simulate_communication_state(fault_impacts, orbital['altitude_km'])
            
            mission_states.append({
                'time_hours': hour,
                'vulnerability_detected': bool(predictions[idx]),
                'vulnerability_score': float(probabilities[idx]),
                'detection_correct': bool(predictions[idx] == labels[idx]),
                'altitude_km': orbital['altitude_km'],
                'velocity_mps': orbital['velocity_mps'],
                'decay_rate': orbital['decay_rate_km_per_day'],
                'power_w': power['power_w'],
                'power_efficiency': power['efficiency'],
                'battery_health': power['battery_health'],
                'temperature_k': thermal['temperature_k'],
                'thermal_margin_k': thermal['thermal_margin_k'],
                'signal_strength_db': comm['signal_strength_db'],
                'data_rate_mbps': comm['data_rate_mbps'],
                'link_availability': comm['link_availability'],
                'affected_subsystems': list(fault_impacts.keys())
            })
        
        return mission_states

    def calculate_mission_success_probability(self, mission_states):
        if not mission_states:
            return 0.0
        
        final = mission_states[-1]
        
        altitude_ok = 1.0 if final['altitude_km'] > 200 else 0.0
        power_ok = final['power_efficiency']
        thermal_ok = 1.0 if final['thermal_margin_k'] > 10 else 0.0
        comm_ok = final['link_availability']
        
        success = (altitude_ok * 0.3 + power_ok * 0.3 + thermal_ok * 0.2 + comm_ok * 0.2)
        
        return success

def export_nasa_results(metrics, mission_states, output_dir="results/nasa_analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mission_states:
        df = pd.DataFrame(mission_states)
        df.to_csv(f"{output_dir}/mission_timeline_{timestamp}.csv", index=False)
    
    report = {
        'detection_performance': metrics['detection_metrics'],
        'vulnerability_analysis': metrics['vulnerability_counts'],
        'nasa_mission_metrics': {
            'risk_score': metrics['mission_risk_score'],
            'assurance_level': metrics['software_assurance_level'],
            'readiness_score': metrics['mission_readiness_score']
        }
    }
    
    with open(f"{output_dir}/nasa_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    return output_dir
