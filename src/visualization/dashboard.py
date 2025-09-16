import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import ast
from mpl_toolkits.axes_grid1 import make_axes_locatable

def create_ksp_dashboard(csv_path, output_path="ksp_mission_dashboard.png"):
    df = pd.read_csv(csv_path)
    
    for col in ['orbital_elements', 'thermal_properties', 'resource_management', 'science_experiments']:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    
    df['apoapsis'] = df['orbital_elements'].apply(lambda x: x['apoapsis'])
    df['periapsis'] = df['orbital_elements'].apply(lambda x: x['periapsis'])
    df['inclination'] = df['orbital_elements'].apply(lambda x: x['inclination'])
    df['eccentricity'] = df['orbital_elements'].apply(lambda x: x['eccentricity'])
    df['temperature'] = df['thermal_properties'].apply(lambda x: x['temperature'])
    df['fuel_percentage'] = df['resource_management'].apply(lambda x: x['fuel_percentage'])
    df['power_level'] = df['resource_management'].apply(lambda x: x['power_level'])
    df['science_multiplier'] = df['science_experiments'].apply(lambda x: x['science_multiplier'])
    
    rocket_cmap = LinearSegmentedColormap.from_list("rocket", 
        ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"])
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 24), facecolor='#0d1117')
    fig.suptitle('KSP Mission Dashboard: Comprehensive Mission Analysis', 
                 fontsize=24, fontweight='bold', color='white', y=0.98)
    
    gs = fig.add_gridspec(4, 2, height_ratios=[1.2, 1, 1, 1], hspace=0.4, wspace=0.3,
                          left=0.06, right=0.96, bottom=0.05, top=0.92)
    
    # 3D Orbital Profile
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.set_facecolor('#0d1117')
    
    u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
    x = 70000 * np.cos(u) * np.sin(v)
    y = 70000 * np.sin(u) * np.sin(v)
    z = 70000 * np.cos(v)
    ax1.plot_surface(x, y, z, color="#2a4d69", alpha=0.3, edgecolor='none')
    
    points = np.array([df.altitude/1000, df.inclination, df.eccentricity*50000]).T
    sc = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                     c=df.velocity_magnitude, cmap=rocket_cmap, s=20, alpha=0.8)
    
    ax1.set_xlabel('Altitude (km)', labelpad=15)
    ax1.set_ylabel('Inclination (°)', labelpad=15)
    ax1.set_zlabel('Eccentricity (scaled)', labelpad=15)
    ax1.set_title('3D Orbital Profile', fontsize=16, fontweight='bold', pad=20)
    
    cbar = plt.colorbar(sc, ax=ax1, shrink=0.5, pad=0.1)
    cbar.set_label('Velocity (m/s)', rotation=270, labelpad=20)
    
    # Resource Radar Chart
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    
    categories = ['Fuel', 'Power', 'Science']
    values = [df.fuel_percentage.iloc[-1]/100, df.power_level.iloc[-1]/100, 
              df.science_multiplier.iloc[-1]/2]
    values += values[:1]
    
    angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
    angles += angles[:1]
    
    ax2.plot(angles, values, linewidth=2, color='#2ecc71')
    ax2.fill(angles, values, alpha=0.25, color='#2ecc71')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.set_title('Resource Status', fontsize=16, fontweight='bold', pad=20)
    
    # Thermal Management
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor('#0d1117')
    ax3.plot(df.temperature, color='#ff9e80', linewidth=2)
    ax3.set_ylabel('Temperature (K)', color='#ff9e80')
    ax3.set_title('Thermal Management', fontsize=16, fontweight='bold')
    
    # Orbital Parameters
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#0d1117')
    
    apo_norm = df.apoapsis / df.apoapsis.max()
    peri_norm = df.periapsis / df.periapsis.max()
    
    ax4.fill_between(df.index, 0, apo_norm, alpha=0.6, label='Apoapsis', color='#3498db')
    ax4.fill_between(df.index, 0, peri_norm, alpha=0.6, label='Periapsis', color='#2ecc71')
    ax4.set_title('Orbital Parameters', fontsize=16, fontweight='bold')
    ax4.legend()
    
    # Mission Health
    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor('#0d1117')
    
    health_score = (df.fuel_percentage + df.power_level) / 2
    ax5.plot(health_score, color='#2ecc71', linewidth=3)
    ax5.set_ylabel('Health Score (%)')
    ax5.set_title('Mission Health Status', fontsize=16, fontweight='bold')
    ax5.axhspan(0, 40, color='#e74c3c', alpha=0.1)
    ax5.axhspan(40, 70, color='#f39c12', alpha=0.1)
    ax5.axhspan(70, 100, color='#2ecc71', alpha=0.1)
    
    # Mission Statistics Footer
    footer_text = f"""Mission Stats: Duration: {len(df)} | Max Alt: {df.altitude.max():.0f}m | Final Fuel: {df.fuel_percentage.iloc[-1]:.1f}%"""
    fig.text(0.5, 0.02, footer_text, ha='center', fontsize=10, alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    
    return output_path