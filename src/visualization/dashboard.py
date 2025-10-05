# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Patch
# import matplotlib.colors as mcolors
# from matplotlib.colors import LinearSegmentedColormap
# import ast
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# def create_ksp_dashboard(csv_path, output_path="ksp_mission_dashboard.png"):
#     df = pd.read_csv(csv_path)
    
#     for col in ['orbital_elements', 'thermal_properties', 'resource_management', 'science_experiments']:
#         if col in df.columns:
#             df[col] = df[col].apply(ast.literal_eval)
    
#     df['apoapsis'] = df['orbital_elements'].apply(lambda x: x['apoapsis'])
#     df['periapsis'] = df['orbital_elements'].apply(lambda x: x['periapsis'])
#     df['inclination'] = df['orbital_elements'].apply(lambda x: x['inclination'])
#     df['eccentricity'] = df['orbital_elements'].apply(lambda x: x['eccentricity'])
#     df['temperature'] = df['thermal_properties'].apply(lambda x: x['temperature'])
#     df['fuel_percentage'] = df['resource_management'].apply(lambda x: x['fuel_percentage'])
#     df['power_level'] = df['resource_management'].apply(lambda x: x['power_level'])
#     df['science_multiplier'] = df['science_experiments'].apply(lambda x: x['science_multiplier'])
    
#     rocket_cmap = LinearSegmentedColormap.from_list("rocket", 
#         ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"])
    
#     plt.style.use('dark_background')
#     fig = plt.figure(figsize=(20, 24), facecolor='#0d1117')
#     fig.suptitle('KSP Mission Dashboard: Comprehensive Mission Analysis', 
#                  fontsize=24, fontweight='bold', color='white', y=0.98)
    
#     gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1], hspace=0.4, wspace=0.3,
#                           left=0.06, right=0.96, bottom=0.05, top=0.92)
    
#     # 3D Orbital Profile
#     ax1 = fig.add_subplot(gs[0, 0], projection='3d')
#     ax1.set_facecolor('#0d1117')
    
#     u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
#     x = 70000 * np.cos(u) * np.sin(v)
#     y = 70000 * np.sin(u) * np.sin(v)
#     z = 70000 * np.cos(v)
#     ax1.plot_surface(x, y, z, color="#2a4d69", alpha=0.3, edgecolor='none')
    
#     points = np.array([df.altitude/1000, df.inclination, df.eccentricity*50000]).T
#     sc = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                      c=df.velocity_magnitude, cmap=rocket_cmap, s=20, alpha=0.8)
    
#     ax1.set_xlabel('Altitude (km)', labelpad=15)
#     ax1.set_ylabel('Inclination (°)', labelpad=15)
#     ax1.set_zlabel('Eccentricity (scaled)', labelpad=15)
#     ax1.set_title('3D Orbital Profile', fontsize=16, fontweight='bold', pad=20)
    
#     cbar = plt.colorbar(sc, ax=ax1, shrink=0.5, pad=0.1)
#     cbar.set_label('Velocity (m/s)', rotation=270, labelpad=20)
    
#     # Resource Radar Chart
#     ax2 = fig.add_subplot(gs[0, 1], polar=True)
    
#     categories = ['Fuel', 'Power', 'Science']
#     values = [df.fuel_percentage.iloc[-1]/100, df.power_level.iloc[-1]/100, 
#               df.science_multiplier.iloc[-1]/2]
#     values += values[:1]
    
#     angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
#     angles += angles[:1]
    
#     ax2.plot(angles, values, linewidth=2, color='#2ecc71')
#     ax2.fill(angles, values, alpha=0.25, color='#2ecc71')
#     ax2.set_xticks(angles[:-1])
#     ax2.set_xticklabels(categories, fontsize=12)
#     ax2.set_ylim(0, 1)
#     ax2.set_title('Resource Status', fontsize=16, fontweight='bold', pad=20)
    
#     # Thermal Management
#     ax3 = fig.add_subplot(gs[1, 0])
#     ax3.set_facecolor('#0d1117')
#     ax3.plot(df.temperature, color='#ff9e80', linewidth=2)
#     ax3.set_ylabel('Temperature (K)', color='#ff9e80')
#     ax3.set_title('Thermal Management', fontsize=16, fontweight='bold')
    
#     # Orbital Parameters
#     ax4 = fig.add_subplot(gs[1, 1])
#     ax4.set_facecolor('#0d1117')
    
#     apo_norm = df.apoapsis / df.apoapsis.max()
#     peri_norm = df.periapsis / df.periapsis.max()
    
#     ax4.fill_between(df.index, 0, apo_norm, alpha=0.6, label='Apoapsis', color='#3498db')
#     ax4.fill_between(df.index, 0, peri_norm, alpha=0.6, label='Periapsis', color='#2ecc71')
#     ax4.set_title('Orbital Parameters', fontsize=16, fontweight='bold')
#     ax4.legend()
    
#     # Mission Health
#     ax5 = fig.add_subplot(gs[2, :])
#     ax5.set_facecolor('#0d1117')
    
#     health_score = (df.fuel_percentage + df.power_level) / 2
#     ax5.plot(health_score, color='#2ecc71', linewidth=3)
#     ax5.set_ylabel('Health Score (%)')
#     ax5.set_title('Mission Health Status', fontsize=16, fontweight='bold')
#     ax5.axhspan(0, 40, color='#e74c3c', alpha=0.1)
#     ax5.axhspan(40, 70, color='#f39c12', alpha=0.1)
#     ax5.axhspan(70, 100, color='#2ecc71', alpha=0.1)
    
#     # Mission Statistics Footer
#     footer_text = f"""Mission Stats: Duration: {len(df)} | Max Alt: {df.altitude.max():.0f}m | Final Fuel: {df.fuel_percentage.iloc[-1]:.1f}%"""
#     fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, alpha=0.7)
    
#     plt.tight_layout(rect=[0, 0.03, 1, 0.98])
#     plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
#     plt.close()
    

#     return output_path


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

def create_nasa_dashboard(metrics, mission_csv_path, output_path="nasa_dashboard.png"):
    df = pd.read_csv(mission_csv_path)
    
    nasa_cmap = LinearSegmentedColormap.from_list(
        "nasa", ["#0B3D91", "#FC3D21", "#FFFFFF"]
    )
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(24, 18), facecolor='#0a0a0a')
    
    fig.suptitle('NASA Spacecraft Software Validation Dashboard', 
                 fontsize=26, fontweight='bold', color='white', y=0.98)
    
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 0.8], 
                          hspace=0.35, wspace=0.3,
                          left=0.05, right=0.97, bottom=0.05, top=0.94)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0a0a0a')
    metrics_data = metrics['detection_metrics']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    metric_values = [
        metrics_data['accuracy'],
        metrics_data['precision'],
        metrics_data['recall'],
        metrics_data['f1_score'],
        metrics_data['auc']
    ]
    
    colors = ['#2ecc71' if v >= 0.9 else '#f39c12' if v >= 0.8 else '#e74c3c' 
              for v in metric_values]
    bars = ax1.barh(metric_names, metric_values, color=colors, edgecolor='white', linewidth=1.5)
    
    for i, (bar, val) in enumerate(zip(bars, metric_values)):
        ax1.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=11, fontweight='bold')
    
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Detection Performance Metrics', fontsize=14, fontweight='bold', pad=15)
    ax1.axvline(x=0.9, color='#2ecc71', linestyle='--', alpha=0.3, linewidth=2)
    ax1.grid(axis='x', alpha=0.2)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#0a0a0a')
    vuln_data = metrics['vulnerability_counts']
    categories = ['Critical', 'High', 'Medium', 'Low']
    counts = [vuln_data['critical'], vuln_data['high'], 
              vuln_data['medium'], vuln_data['low']]
    severity_colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db']
    
    wedges, texts, autotexts = ax2.pie(
        counts, labels=categories, colors=severity_colors,
        autopct='%1.1f%%', startangle=90, 
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    ax2.set_title(f'Vulnerability Distribution (Total: {vuln_data["total"]})', 
                  fontsize=14, fontweight='bold', pad=15)
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#0a0a0a')
    ax3.axis('off')
    
    risk_score = metrics['mission_risk_score']
    assurance = metrics['software_assurance_level']
    readiness = metrics['mission_readiness_score']
    
    y_pos = 0.85
    title_props = {'fontsize': 16, 'fontweight': 'bold', 'color': 'white'}
    value_props = {'fontsize': 14, 'fontweight': 'bold'}
    
    ax3.text(0.5, y_pos, 'NASA Mission Metrics', ha='center', **title_props)
    
    risk_color = '#2ecc71' if risk_score < 0.3 else '#f39c12' if risk_score < 0.6 else '#e74c3c'
    ax3.text(0.1, y_pos - 0.2, 'Risk Score:', fontsize=13, color='#aaa')
    ax3.text(0.9, y_pos - 0.2, f'{risk_score:.3f}', ha='right', color=risk_color, **value_props)
    
    assurance_color = {'EXCELLENT': '#2ecc71', 'ACCEPTABLE': '#3498db', 
                       'MARGINAL': '#f39c12', 'REQUIRES_REMEDIATION': '#e74c3c'}
    ax3.text(0.1, y_pos - 0.35, 'Assurance:', fontsize=13, color='#aaa')
    ax3.text(0.9, y_pos - 0.35, assurance, ha='right', 
             color=assurance_color.get(assurance, 'white'), **value_props)
    
    readiness_color = '#2ecc71' if readiness > 0.8 else '#f39c12' if readiness > 0.6 else '#e74c3c'
    ax3.text(0.1, y_pos - 0.5, 'Readiness:', fontsize=13, color='#aaa')
    ax3.text(0.9, y_pos - 0.5, f'{readiness:.3f}', ha='right', 
             color=readiness_color, **value_props)
    
    rect = FancyBboxPatch((0.05, 0.05), 0.9, 0.75, 
                          boxstyle="round,pad=0.02", 
                          edgecolor='#0B3D91', facecolor='none', 
                          linewidth=3, transform=ax3.transAxes)
    ax3.add_patch(rect)
    
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_facecolor('#0a0a0a')
    
    ax4_alt = ax4.twinx()
    ax4_pow = ax4.twinx()
    ax4_pow.spines['right'].set_position(('outward', 60))
    
    ln1 = ax4.plot(df['time_hours'], df['altitude_km'], 
                   color='#3498db', linewidth=2.5, label='Altitude', alpha=0.9)
    ln2 = ax4_alt.plot(df['time_hours'], df['temperature_k'], 
                       color='#e74c3c', linewidth=2.5, label='Temperature', alpha=0.9)
    ln3 = ax4_pow.plot(df['time_hours'], df['power_w'], 
                       color='#2ecc71', linewidth=2.5, label='Power', alpha=0.9)
    
    ax4.set_xlabel('Mission Time (hours)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Altitude (km)', color='#3498db', fontsize=12, fontweight='bold')
    ax4_alt.set_ylabel('Temperature (K)', color='#e74c3c', fontsize=12, fontweight='bold')
    ax4_pow.set_ylabel('Power (W)', color='#2ecc71', fontsize=12, fontweight='bold')
    
    ax4.tick_params(axis='y', labelcolor='#3498db')
    ax4_alt.tick_params(axis='y', labelcolor='#e74c3c')
    ax4_pow.tick_params(axis='y', labelcolor='#2ecc71')
    
    ax4.set_title('Mission Timeline: Critical Parameters', fontsize=14, fontweight='bold', pad=15)
    
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, loc='upper right', framealpha=0.9)
    ax4.grid(alpha=0.2)
    
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor('#0a0a0a')
    
    vuln_times = df[df['vulnerability_detected'] == True]['time_hours']
    if len(vuln_times) > 0:
        ax5.hist(vuln_times, bins=30, color='#e74c3c', alpha=0.7, edgecolor='white')
    
    ax5.set_xlabel('Mission Time (hours)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Vulnerability Count', fontsize=12, fontweight='bold')
    ax5.set_title('Vulnerability Detection Timeline', fontsize=14, fontweight='bold', pad=15)
    ax5.grid(alpha=0.2)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor('#0a0a0a')
    
    window = max(1, len(df) // 50)
    df['power_ma'] = df['power_w'].rolling(window=window, center=True).mean()
    df['efficiency_ma'] = df['power_efficiency'].rolling(window=window, center=True).mean()
    
    ax6.fill_between(df['time_hours'], 0, df['power_efficiency'] * 100, 
                     alpha=0.3, color='#2ecc71', label='Efficiency')
    ax6.plot(df['time_hours'], df['battery_health'] * 100, 
             color='#f39c12', linewidth=2, label='Battery Health')
    
    ax6.set_xlabel('Mission Time (hours)', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax6.set_title('Power System Health', fontsize=14, fontweight='bold', pad=15)
    ax6.legend(loc='best', framealpha=0.9)
    ax6.grid(alpha=0.2)
    ax6.set_ylim(0, 105)
    
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor('#0a0a0a')
    
    ax7.plot(df['time_hours'], df['signal_strength_db'], 
             color='#9b59b6', linewidth=2, label='Signal Strength')
    ax7.axhline(y=-100, color='#e74c3c', linestyle='--', alpha=0.5, label='Critical Threshold')
    
    ax7_rate = ax7.twinx()
    ax7_rate.plot(df['time_hours'], df['data_rate_mbps'], 
                  color='#1abc9c', linewidth=2, label='Data Rate', alpha=0.8)
    
    ax7.set_xlabel('Mission Time (hours)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Signal Strength (dB)', color='#9b59b6', fontsize=12, fontweight='bold')
    ax7_rate.set_ylabel('Data Rate (Mbps)', color='#1abc9c', fontsize=12, fontweight='bold')
    
    ax7.tick_params(axis='y', labelcolor='#9b59b6')
    ax7_rate.tick_params(axis='y', labelcolor='#1abc9c')
    
    ax7.set_title('Communication Performance', fontsize=14, fontweight='bold', pad=15)
    
    lines1, labels1 = ax7.get_legend_handles_labels()
    lines2, labels2 = ax7_rate.get_legend_handles_labels()
    ax7.legend(lines1 + lines2, labels1 + labels2, loc='best', framealpha=0.9)
    ax7.grid(alpha=0.2)
    
    ax8 = fig.add_subplot(gs[3, :])
    ax8.set_facecolor('#0a0a0a')
    ax8.axis('off')
    
    final_state = df.iloc[-1]
    
    summary_metrics = [
        ('Final Altitude', f"{final_state['altitude_km']:.1f} km", 
         '#3498db' if final_state['altitude_km'] > 200 else '#e74c3c'),
        ('Final Velocity', f"{final_state['velocity_mps']:.1f} m/s", '#3498db'),
        ('Power Efficiency', f"{final_state['power_efficiency']*100:.1f}%", 
         '#2ecc71' if final_state['power_efficiency'] > 0.7 else '#f39c12'),
        ('Thermal Margin', f"{final_state['thermal_margin_k']:.1f} K", 
         '#2ecc71' if final_state['thermal_margin_k'] > 10 else '#e74c3c'),
        ('Link Availability', f"{final_state['link_availability']*100:.1f}%", 
         '#2ecc71' if final_state['link_availability'] > 0.8 else '#f39c12'),
        ('Detected Vulnerabilities', f"{df['vulnerability_detected'].sum()}", '#e74c3c'),
        ('Correct Detections', f"{df['detection_correct'].sum()}", '#2ecc71'),
        ('Mission Duration', f"{len(df)} hours", '#3498db')
    ]
    
    cols = 4
    rows = 2
    for i, (label, value, color) in enumerate(summary_metrics):
        row = i // cols
        col = i % cols
        x = 0.05 + (col * 0.24)
        y = 0.65 - (row * 0.35)
        
        box = FancyBboxPatch((x, y), 0.20, 0.25,
                            boxstyle="round,pad=0.01",
                            edgecolor=color, facecolor='#1a1a1a',
                            linewidth=2, transform=ax8.transAxes)
        ax8.add_patch(box)
        
        ax8.text(x + 0.10, y + 0.18, label, ha='center', va='center',
                fontsize=10, color='#aaa', transform=ax8.transAxes)
        ax8.text(x + 0.10, y + 0.08, value, ha='center', va='center',
                fontsize=13, fontweight='bold', color=color, transform=ax8.transAxes)
    
    footer_text = (
        f"NASA Software Assurance Standard: NASA-STD-8739.8 | "
        f"Mission Risk: {metrics['mission_risk_score']:.3f} | "
        f"Readiness: {metrics['mission_readiness_score']:.3f}"
    )
    fig.text(0.5, 0.01, footer_text, ha='center', fontsize=11, 
             alpha=0.8, color='#0B3D91', fontweight='bold')
    
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0a', bbox_inches='tight')
    plt.close()
    
    return output_path
