import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

# Initialize session state
def init_session_state():
    if 'targets' not in st.session_state:
        st.session_state.targets = pd.DataFrame({
            'id': ['T1'],
            'range': [180.0],
            'bearing': [45.0],
            'elevation': [5.0],
            'velocity': [-0.03],
            'threat_level': [0.7],
            'status': ['Tracking'],
            'x': [180.0 * np.cos(np.radians(45.0))],
            'y': [180.0 * np.sin(np.radians(45.0))]
        })
        
    if 'interceptors' not in st.session_state:
        st.session_state.interceptors = pd.DataFrame({
            'id': [f'I{i+1}' for i in range(50)],
            'speed': np.random.uniform(0.08, 0.15, 50),
            'range_capability': np.random.uniform(120, 200, 50),
            'status': ['Available'] * 50,
            'range': [0.0] * 50,
            'bearing': [0.0] * 50,
            'elevation': [0.0] * 50,
            'x': [0.0] * 50,
            'y': [0.0] * 50,
            'target_id': [''] * 50,
            'engaged_until': [0.0] * 50
        })
        
    if 'command_log' not in st.session_state:
        st.session_state.command_log = []
        
    if 'last_spawn_time' not in st.session_state:
        st.session_state.last_spawn_time = time.time()
        
    if 'sweep_angle' not in st.session_state:
        st.session_state.sweep_angle = 0
        
    if 'neutralization_flash' not in st.session_state:
        st.session_state.neutralization_flash = {}
        
    if 'display_mode' not in st.session_state:
        st.session_state.display_mode = 'All Drones'
        
    if 'plot_axes' not in st.session_state:
        st.session_state.plot_axes = {}
        
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
        
    if 'last_update_time' not in st.session_state:
        st.session_state.last_update_time = time.time()

# Spawn drone swarms
def spawn_drones():
    targets = st.session_state.targets
    if len(targets) >= 25:
        return
        
    if time.time() - st.session_state.last_spawn_time < np.random.uniform(5, 10):
        return
        
    num_drones = np.random.randint(5, 9)
    for _ in range(min(num_drones, 25 - len(targets))):
        new_id = f'T{len(targets) + 1}'
        new_range = np.random.uniform(150, 200)
        new_bearing = np.random.uniform(0, 360)
        new_elevation = np.random.uniform(0, 10)
        new_velocity = np.random.uniform(-0.05, -0.01)
        threat_level = 0.5 * (abs(new_velocity) / 0.05) + 0.3 * (200 - new_range) / 200 + 0.2 * (new_elevation / 10)
        new_target = pd.DataFrame({
            'id': [new_id],
            'range': [new_range],
            'bearing': [new_bearing],
            'elevation': [new_elevation],
            'velocity': [new_velocity],
            'threat_level': [threat_level],
            'status': ['Tracking'],
            'x': [new_range * np.cos(np.radians(new_bearing))],
            'y': [new_range * np.sin(np.radians(new_bearing))]
        })
        st.session_state.targets = pd.concat([st.session_state.targets, new_target], ignore_index=True)
        if threat_level >= 0.8:
            st.session_state.targets.loc[st.session_state.targets['id'] == new_id, 'status'] = 'Prioritized'
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Drone {new_id} auto-prioritized (Threat: {threat_level:.2f})")
    st.session_state.last_spawn_time = time.time()

# Update drone positions
def update_drones():
    targets = st.session_state.targets
    targets['range'] = targets['range'] + targets['velocity']
    targets['x'] = targets['range'] * np.cos(np.radians(targets['bearing']))
    targets['y'] = targets['range'] * np.sin(np.radians(targets['bearing']))
    targets.loc[(targets['range'] < 20) & (targets['status'] != 'Neutralized'), 'status'] = 'Escaped'
    targets.loc[targets['range'] < 0, 'range'] = 0
    targets.loc[targets['range'] > 200, 'range'] = 200
    targets['elevation'] = targets['elevation'].clip(0, 10)
    st.session_state.targets = targets

# Update interceptor positions
def update_interceptors():
    interceptors = st.session_state.interceptors
    targets = st.session_state.targets
    current_time = time.time()
    engaged = interceptors[interceptors['status'] != 'Available']
    
    for idx, interceptor in engaged.iterrows():
        if current_time >= interceptor['engaged_until']:
            interceptors.loc[interceptors['id'] == interceptor['id'], 
                             ['status', 'range', 'bearing', 'elevation', 'x', 'y', 'target_id', 'engaged_until']] = \
                ['Available', 0, 0, 0, 0, 0, '', 0]
            continue
                
        target = targets[targets['id'] == interceptor['target_id']]
        if target.empty or target['status'].iloc[0] in ['Neutralized', 'Escaped']:
            continue
            
        target_x, target_y, target_elevation = target['x'].iloc[0], target['y'].iloc[0], target['elevation'].iloc[0]
        current_x, current_y, current_elevation = interceptor['x'], interceptor['y'], interceptor['elevation']
        distance = np.sqrt((target_x - current_x)**2 + (target_y - current_y)**2 + (target_elevation - current_elevation)**2)
        
        if distance < 0.1:  # Neutralization threshold
            targets.loc[targets['id'] == interceptor['target_id'], 'status'] = 'Neutralized'
            st.session_state.neutralization_flash[interceptor['target_id']] = current_time
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: ALERT: Interceptor {interceptor['id']} neutralized {interceptor['target_id']} at {interceptor['range']:.2f} km [BOOM]")
            interceptors.loc[interceptors['id'] == interceptor['id'], 
                             ['status', 'range', 'bearing', 'elevation', 'x', 'y', 'target_id']] = \
                ['Engaged', interceptor['range'], interceptor['bearing'], target_elevation, target_x, target_y, '']
            continue
                
        direction_x = (target_x - current_x) / distance
        direction_y = (target_y - current_y) / distance
        direction_z = (target_elevation - current_elevation) / distance
        new_x = current_x + interceptor['speed'] * direction_x
        new_y = current_y + interceptor['speed'] * direction_y
        new_elevation = current_elevation + interceptor['speed'] * direction_z
        new_range = np.sqrt(new_x**2 + new_y**2)
        new_bearing = np.degrees(np.arctan2(new_y, new_x)) % 360
        
        interceptors.loc[interceptors['id'] == interceptor['id'], 
                         ['x', 'y', 'elevation', 'range', 'bearing']] = \
            [new_x, new_y, new_elevation, new_range, new_bearing]
            
        if distance < 1:
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Interceptor {interceptor['id']} locked on {interceptor['target_id']} (Distance: {distance:.2f} km)")
    
    st.session_state.interceptors = interceptors
    st.session_state.targets = targets

# Prioritize drones
def prioritize_drones():
    targets = st.session_state.targets
    for idx, target in targets.iterrows():
        if target['status'] in ['Neutralized', 'Escaped']:
            continue
            
        time_to_target = target['range'] / abs(target['velocity'])
        nn_score = 0.4 * target['threat_level'] + 0.3 * (30 / max(time_to_target, 1)) + \
                   0.2 * (abs(target['velocity']) / 0.05) + 0.1 * (target['elevation'] / 10)
                    
        if nn_score >= 0.8:
            targets.loc[targets['id'] == target['id'], 'status'] = 'Prioritized'
    
    st.session_state.targets = targets

# Assign interceptors
def assign_interceptors(auto_all=False):
    targets = st.session_state.targets
    interceptors = st.session_state.interceptors
    current_time = time.time()
    
    interceptors.loc[interceptors['engaged_until'] <= current_time, 
                     ['status', 'range', 'bearing', 'elevation', 'x', 'y', 'target_id', 'engaged_until']] = \
        ['Available', 0, 0, 0, 0, 0, '', 0]
        
    prioritized = targets[targets['status'] == 'Prioritized'].sort_values(by='threat_level', ascending=False)
    available_interceptors = interceptors[interceptors['status'] == 'Available']

    for _, target in prioritized.iterrows():
        if available_interceptors.empty:
            if auto_all:
                st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: No interceptors available for {target['id']}")
            break
            
        if target['range'] < 20:
            continue
            
        target_range = target['range']
        target_velocity = abs(target['velocity'])
        valid_interceptors = available_interceptors[available_interceptors['range_capability'] >= target_range]
        
        if valid_interceptors.empty:
            if auto_all:
                st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: No suitable interceptor for {target['id']} (range {target_range:.2f} km)")
            continue
            
        scores = 0.5 * (valid_interceptors['speed'] / target_velocity) + \
                 0.3 * (1 - abs(valid_interceptors['range_capability'] - target_range) / 200) + \
                 0.2 * (valid_interceptors['engaged_until'] == 0).astype(int)
                 
        interceptor = valid_interceptors.loc[scores.idxmax()]
        interceptor_id = interceptor['id']
        target_id = target['id']
        confidence = np.random.uniform(0.90, 0.98)
        
        interceptors.loc[interceptors['id'] == interceptor_id, 
                         ['status', 'range', 'bearing', 'elevation', 'x', 'y', 'target_id', 'engaged_until']] = \
            [f'Engaged with {target_id}', target_range, target['bearing'], target['elevation'], 
             target['x'], target['y'], target_id, current_time + 10]
             
        available_interceptors = interceptors[interceptors['status'] == 'Available']
        
        if auto_all:
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Interceptor {interceptor_id} assigned to {target_id} (Confidence: {confidence:.2%})")
    
    st.session_state.interceptors = interceptors

# Manual interceptor assignment
def manual_assign_interceptor(target_id, ml_model):
    targets = st.session_state.targets
    interceptors = st.session_state.interceptors
    target = targets[targets['id'] == target_id]
    
    if target.empty or target['status'].iloc[0] in ['Neutralized', 'Escaped']:
        result = f"Error: Cannot assign interceptor to {target_id}. Invalid target status."
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
        return result
        
    target_range = target['range'].iloc[0]
    if target_range < 20:
        result = f"Error: Drone {target_id} too close ({target_range:.2f} km) to intercept."
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
        return result
        
    available_interceptors = interceptors[interceptors['status'] == 'Available']
    if available_interceptors.empty:
        result = "Error: No interceptors available."
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
        return result
        
    target_velocity = abs(target['velocity'].iloc[0])
    valid_interceptors = available_interceptors[available_interceptors['range_capability'] >= target_range]
    
    if valid_interceptors.empty:
        result = f"Error: No suitable interceptor for {target_id} (range {target_range:.2f} km)"
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
        return result
        
    if ml_model == 'Logistic Regression':
        scores = 1 - abs(valid_interceptors['range_capability'] - target_range) / 200
        confidence = 0.85
    elif ml_model == 'Random Forest':
        scores = valid_interceptors['speed'] / target_velocity
        confidence = 0.90
    else:  # Neural Network
        scores = 0.4 * (valid_interceptors['speed'] / target_velocity) + \
                 0.3 * (1 - abs(valid_interceptors['range_capability'] - target_range) / 200) + \
                 0.3 * (valid_interceptors['elevation'] / 10)
        confidence = 0.95
        
    interceptor = valid_interceptors.loc[scores.idxmax()]
    interceptor_id = interceptor['id']
    
    interceptors.loc[interceptors['id'] == interceptor_id, 
                     ['status', 'range', 'bearing', 'elevation', 'x', 'y', 'target_id', 'engaged_until']] = \
        [f'Engaged with {target_id}', target_range, target['bearing'].iloc[0], target['elevation'].iloc[0],
         target['x'].iloc[0], target['y'].iloc[0], target_id, time.time() + 10]
         
    result = f"Interceptor {interceptor_id} assigned to {target_id} using {ml_model} (Confidence: {confidence:.2%})"
    st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: {result}")
    return result

# Create tactical plots
def create_tactical_plots():
    fig = plt.figure(figsize=(20, 10), facecolor='#0a0a23')
    plot_axes = {}
    plot_axes['radar'] = fig.add_subplot(121, projection='polar', facecolor='#1a1a3a')
    plot_axes['geo'] = fig.add_subplot(122, facecolor='#1a1a3a')
    plot_axes['status'] = fig.add_axes([0.1, 0.05, 0.8, 0.1], facecolor='#1a1a3a')
    ax_inset = None

    # Radar Plot
    plot_axes['radar'].set_ylim(0, 200)
    plot_axes['radar'].set_yticks([20, 50, 100, 150, 200])
    plot_axes['radar'].set_yticklabels(['20km', '50km', '100km', '150km', '200km'], color='#00ffcc', fontsize=12)
    plot_axes['radar'].set_xticks(np.radians(np.arange(0, 360, 30)))
    plot_axes['radar'].set_xticklabels(['N', '30', '60', 'E', '120', '150', 'S', '210', '240', 'W', '300', '330'],
                                       color='#00ffcc', fontsize=12)
    plot_axes['radar'].set_theta_zero_location('N')
    plot_axes['radar'].set_theta_direction(-1)
    plot_axes['radar'].grid(color='#00ffcc', linestyle='--', alpha=0.3)
    plot_axes['radar'].fill_between(np.radians([90, 150]), 0, 200, color='#ff3333', alpha=0.1, label='High Threat Zone')
    plot_axes['radar'].plot(np.radians(np.linspace(0, 360, 100)), [20]*100, color='#ff3333', linestyle='--', alpha=0.5, label='20km Boundary')

    # Radar sweep
    sweep_start = st.session_state.sweep_angle % 360
    sweep_end = (st.session_state.sweep_angle + 45) % 360
    if sweep_start < sweep_end:
        angles = np.radians(np.linspace(sweep_start, sweep_end, 50))
    else:
        angles = np.radians(np.linspace(sweep_start, 360, 25).tolist() + np.linspace(0, sweep_end, 25).tolist())
    plot_axes['radar'].fill_between(angles, 0, 200, color='#00ff00', alpha=0.3, label='Doppler Sweep')
    st.session_state.sweep_angle = (st.session_state.sweep_angle + 90) % 360

    # Threat heatmap
    for _, row in st.session_state.targets.iterrows():
        if row['range'] < 50 or row['threat_level'] >= 0.9:
            plot_axes['radar'].scatter(np.radians(row['bearing']), row['range'], c='none', s=500, marker='o',
                                       edgecolors='#ff3333', alpha=0.2)

    # Determine display targets
    selected_target = st.session_state.selected_target
    if st.session_state.display_mode == 'Selected Target' and selected_target and selected_target in st.session_state.targets['id'].values:
        display_targets = st.session_state.targets[st.session_state.targets['id'] == selected_target]
    else:
        display_targets = st.session_state.targets[st.session_state.targets['status'].isin(['Tracking', 'Prioritized'])]

    # Plot drones
    current_time = time.time()
    data_feed = []
    for _, row in display_targets.iterrows():
        color = '#ff3333' if row['threat_level'] >= 0.8 else '#ff9900' if row['threat_level'] >= 0.6 else '#ffff00'
        size = 150 + 100 * row['elevation'] / 10
        plot_axes['radar'].scatter(np.radians(row['bearing']), row['range'], c=color, s=size, marker='o', alpha=0.9)
        plot_axes['radar'].text(np.radians(row['bearing']), row['range'], f"{row['id']} ({row['elevation']:.1f}km)",
                                fontsize=10, color='#ffffff', ha='center', va='bottom', fontweight='bold')
        if row['status'] not in ['Neutralized', 'Escaped']:
            ranges = np.linspace(row['range'], row['range'] - 15 * row['velocity'], 15)
            bearings = np.full(15, np.radians(row['bearing']))
            plot_axes['radar'].plot(bearings, ranges, color=color, linestyle='--', alpha=0.5)
        if row['status'] == 'Neutralized' and row['id'] in st.session_state.neutralization_flash:
            if current_time - st.session_state.neutralization_flash[row['id']] < 3:
                plot_axes['radar'].scatter(np.radians(row['bearing']), row['range'], c='none', s=600, marker='o',
                                           edgecolors='#ff3333', linewidth=3, alpha=0.8)
                plot_axes['radar'].scatter(np.radians(row['bearing']), row['range'], c='#ffff00', s=300, marker='*', alpha=0.7)
            else:
                st.session_state.neutralization_flash.pop(row['id'], None)

    # Plot interceptors
    engaged = st.session_state.interceptors[st.session_state.interceptors['status'] != 'Available']
    if st.session_state.display_mode == 'Selected Target' and selected_target:
        engaged = engaged[engaged['target_id'] == selected_target]
    if not engaged.empty:
        for _, row in engaged.iterrows():
            size = 100 + 50 * row['elevation'] / 10
            plot_axes['radar'].scatter(np.radians(row['bearing']), row['range'], c='#00ccff', s=size, marker='^', alpha=0.9)
            plot_axes['radar'].text(np.radians(row['bearing']), row['range'], f"{row['id']} ({row['elevation']:.1f}km)",
                                    fontsize=10, color='#ffffff', ha='center', va='bottom', fontweight='bold')
            if row['target_id']:
                target = st.session_state.targets[st.session_state.targets['id'] == row['target_id']]
                if not target.empty and target['status'].iloc[0] not in ['Neutralized', 'Escaped']:
                    ranges = np.linspace(row['range'], target['range'].iloc[0], 15)
                    bearings = np.full(15, np.radians(row['bearing']))
                    plot_axes['radar'].plot(bearings, ranges, color='#00ccff', linestyle='--', alpha=0.5)
                    distance = np.sqrt((row['x'] - target['x'].iloc[0])**2 + (row['y'] - target['y'].iloc[0])**2)
                    time_to_intercept = distance / row['speed']
                    prob = 0.9 if distance < 5 else 0.7 if distance < 10 else 0.5
                    data_feed.append(f"{row['id']} -> {row['target_id']}: {distance:.2f} km, {time_to_intercept:.1f} s, Prob: {prob:.0%}")

    # Inset plot
    if st.session_state.display_mode == 'Selected Target' and selected_target and selected_target in st.session_state.targets['id'].values:
        target = st.session_state.targets[st.session_state.targets['id'] == selected_target]
        if not target.empty:
            ax_inset = fig.add_axes([0.35, 0.6, 0.15, 0.15], projection='polar', facecolor='#1a1a3a')
            ax_inset.set_ylim(target['range'].iloc[0] - 5, target['range'].iloc[0] + 5)
            ax_inset.set_xticks([])
            ax_inset.set_yticks([])
            ax_inset.scatter(np.radians(target['bearing'].iloc[0]), target['range'].iloc[0], c='#ff3333', s=100, marker='o')
            engaged = st.session_state.interceptors[st.session_state.interceptors['target_id'] == selected_target]
            if not engaged.empty:
                ax_inset.scatter(np.radians(engaged['bearing']), engaged['range'], c='#00ccff', s=50, marker='^')

    plot_axes['radar'].set_title('3D Tactical Radar Display', color='#00ffcc', fontsize=16, pad=20, fontweight='bold')
    plot_axes['radar'].legend(loc='upper right', fontsize=10, facecolor='#1a1a3a', edgecolor='#00ffcc', labelcolor='#00ffcc')

    # Geographic Plot
    plot_axes['geo'].set_xlim(-200, 200)
    plot_axes['geo'].set_ylim(-200, 200)
    plot_axes['geo'].set_xlabel('X (km)', color='#00ffcc', fontsize=12)
    plot_axes['geo'].set_ylabel('Y (km)', color='#00ffcc', fontsize=12)
    plot_axes['geo'].grid(color='#00ffcc', linestyle='--', alpha=0.3)
    plot_axes['geo'].set_facecolor('#1a1a3a')
    circle = plt.Circle((0, 0), 20, color='#ff3333', fill=False, linestyle='--', alpha=0.5)
    plot_axes['geo'].add_patch(circle)
    plot_axes['geo'].contour(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50),
                             np.random.randn(50, 50), levels=3, colors='#00ffcc', alpha=0.2)

    for _, row in display_targets.iterrows():
        color = '#ff3333' if row['threat_level'] >= 0.8 else '#ff9900' if row['threat_level'] >= 0.6 else '#ffff00'
        size = 150 + 100 * row['elevation'] / 10
        plot_axes['geo'].scatter(row['x'], row['y'], c=color, s=size, marker='o', alpha=0.9)
        plot_axes['geo'].text(row['x'], row['y'], f"{row['id']} ({row['elevation']:.1f}km)", fontsize=10, color='#ffffff',
                              ha='center', va='bottom', fontweight='bold')
        if row['status'] not in ['Neutralized', 'Escaped']:
            xs = np.linspace(row['x'], row['x'] - 15 * row['velocity'] * np.cos(np.radians(row['bearing'])), 15)
            ys = np.linspace(row['y'], row['y'] - 15 * row['velocity'] * np.sin(np.radians(row['bearing'])), 15)
            plot_axes['geo'].plot(xs, ys, color=color, linestyle='--', alpha=0.5)
        if row['status'] == 'Neutralized' and row['id'] in st.session_state.neutralization_flash:
            if current_time - st.session_state.neutralization_flash[row['id']] < 3:
                plot_axes['geo'].scatter(row['x'], row['y'], c='none', s=600, marker='o', edgecolors='#ff3333', linewidth=3, alpha=0.8)
                plot_axes['geo'].scatter(row['x'], row['y'], c='#ffff00', s=300, marker='*', alpha=0.7)

    if not engaged.empty:
        for _, row in engaged.iterrows():
            size = 100 + 50 * row['elevation'] / 10
            plot_axes['geo'].scatter(row['x'], row['y'], c='#00ccff', s=size, marker='^', alpha=0.9)
            plot_axes['geo'].text(row['x'], row['y'], f"{row['id']} ({row['elevation']:.1f}km)", fontsize=10, color='#ffffff',
                                  ha='center', va='bottom', fontweight='bold')
            if row['target_id']:
                target = st.session_state.targets[st.session_state.targets['id'] == row['target_id']]
                if not target.empty and target['status'].iloc[0] not in ['Neutralized', 'Escaped']:
                    xs = np.linspace(row['x'], target['x'].iloc[0], 15)
                    ys = np.linspace(row['y'], target['y'].iloc[0], 15)
                    plot_axes['geo'].plot(xs, ys, color='#00ccff', linestyle='--', alpha=0.5)
                    prob = 0.9 if distance < 5 else 0.7 if distance < 10 else 0.5
                    plot_axes['geo'].scatter(row['x'], row['y'], c='none', s=200, marker='o',
                                             edgecolors='#00ff00' if prob > 0.8 else '#ffff00', alpha=0.3)

    plot_axes['geo'].set_title('Satellite Geographic Map', color='#00ffcc', fontsize=16, pad=20, fontweight='bold')

    # Status Dashboard
    tracking = len(st.session_state.targets[st.session_state.targets['status'] == 'Tracking'])
    prioritized = len(st.session_state.targets[st.session_state.targets['status'] == 'Prioritized'])
    neutralized = len(st.session_state.targets[st.session_state.targets['status'] == 'Neutralized'])
    escaped = len(st.session_state.targets[st.session_state.targets['status'] == 'Escaped'])
    available = len(st.session_state.interceptors[st.session_state.interceptors['status'] == 'Available'])
    categories = ['Tracking', 'Prioritized', 'Neutralized', 'Escaped', 'Available']
    values = [tracking, prioritized, neutralized, escaped, available]
    colors = ['#ffff00', '#ff9900', '#00ffcc', '#ff3333', '#00ccff']
    bars = plot_axes['status'].barh(categories, values, color=colors, edgecolor='#00ffcc', alpha=0.8)
    for bar in bars:
        bar.set_alpha(0.8 + 0.2 * np.sin(current_time % 2 * np.pi))
    plot_axes['status'].set_xlim(0, max(max(values, default=1), 50))
    plot_axes['status'].set_facecolor('#1a1a3a')
    plot_axes['status'].tick_params(axis='both', colors='#00ffcc', labelsize=10)
    plot_axes['status'].set_title('Tactical Status Dashboard', color='#00ffcc', fontsize=12)

    if data_feed:
        plot_axes['status'].text(1.05, 0.5, '\n'.join(data_feed[:3]), transform=plot_axes['status'].transAxes, color='#ffffff',
                                fontsize=8, va='center', ha='left', bbox=dict(facecolor='#1a1a3a', edgecolor='#00ffcc', alpha=0.8))

    plt.tight_layout()
    return fig

# Update UI
def update_ui():
    targets = st.session_state.targets
    interceptors = st.session_state.interceptors
    
    if 'selected_target' not in st.session_state:
        st.session_state.selected_target = targets['id'].iloc[0] if not targets.empty else None
    
    # Update widgets
    st.session_state.target_options = targets['id'].tolist()
    
    # Update engaged interceptors data feed
    data_feed = []
    engaged = interceptors[interceptors['status'] != 'Available']
    for _, row in engaged.iterrows():
        if row['target_id']:
            target = targets[targets['id'] == row['target_id']]
            if not target.empty and target['status'].iloc[0] not in ['Neutralized', 'Escaped']:
                distance = np.sqrt((row['x'] - target['x'].iloc[0])**2 + (row['y'] - target['y'].iloc[0])**2)
                time_to_intercept = distance / row['speed']
                prob = 0.9 if distance < 5 else 0.7 if distance < 10 else 0.5
                data_feed.append(f"{row['id']} -> {row['target_id']}: Dist: {distance:.2f} km, Time: {time_to_intercept:.1f} s, Prob: {prob:.0%}")
    
    st.session_state.data_feed = data_feed

# Run simulation step
def run_simulation_step():
    spawn_drones()
    update_drones()
    prioritize_drones()
    update_interceptors()
    assign_interceptors()

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Quantum Radar Mission Simulator",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        :root {
            --primary: #00ffcc;
            --secondary: #cc00ff;
            --bg-dark: #0a0a23;
            --panel-bg: #1a1a3a;
            --warning: #ff9900;
            --danger: #ff3333;
            --success: #00ccff;
        }
        
        body {
            background-color: var(--bg-dark);
            color: var(--primary);
            font-family: 'Courier New', monospace;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0a23 0%, #1a1a3a 100%);
        }
        
        .stButton>button {
            background-color: var(--panel-bg);
            color: var(--primary);
            border: 2px solid var(--primary);
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 0 10px var(--primary);
        }
        
        .stButton>button:hover {
            background-color: var(--primary);
            color: var(--bg-dark);
            box-shadow: 0 0 15px var(--primary);
        }
        
        .stSelectbox>div>div {
            background-color: var(--panel-bg);
            color: var(--primary);
            border: 2px solid var(--primary);
        }
        
        .stSlider>div>div>div {
            background-color: var(--primary);
        }
        
        .stTextInput>div>div>input {
            background-color: var(--panel-bg);
            color: var(--primary);
            border: 2px solid var(--primary);
        }
        
        .stDataFrame {
            background-color: var(--panel-bg);
            border: 2px solid var(--primary);
            border-radius: 5px;
        }
        
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: var(--primary);
            text-shadow: 0 0 10px var(--primary);
        }
        
        .panel {
            background-color: var(--panel-bg);
            border: 2px solid var(--primary);
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 0 15px var(--primary);
        }
        
        .log-panel {
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            background-color: #0a0a23;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid var(--primary);
        }
        
        .data-feed {
            background-color: #0a0a23;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid var(--success);
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Header
    st.title("Quantum Radar Mission Simulator")
    st.markdown("---")
    
    # Control Panel
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.session_state.selected_target = st.selectbox(
                "Select Target", 
                st.session_state.targets['id'].tolist() if not st.session_state.targets.empty else [],
                key='target_selector'
            )
            
        with col2:
            command = st.selectbox(
                "Command", 
                ["Track", "Prioritize", "Intercept"],
                key='command_selector'
            )
            
        with col3:
            st.session_state.ml_model = st.selectbox(
                "ML Model", 
                ["Logistic Regression", "Random Forest", "Neural Network"],
                key='ml_selector'
            )
            
        with col4:
            st.write("")
            st.write("")
            execute_btn = st.button("Execute", key='execute_btn')
            
    # Action Buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    
    with col1:
        track_all_btn = st.button("Track All", key='track_all_btn')
        
    with col2:
        assign_btn = st.button("Manual Assign", key='assign_btn')
        
    with col3:
        auto_assign_btn = st.button("Auto Assign All", key='auto_assign_btn')
        
    with col4:
        st.session_state.display_mode = st.radio(
            "Display Mode",
            ["All Drones", "Selected Target"],
            horizontal=True,
            key='display_selector'
        )
    
    # Simulation Control - UPDATED SECTION
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.session_state.simulation_running:
            if st.button("Stop Simulation"):
                st.session_state.simulation_running = False
        else:
            if st.button("Start Simulation"):
                st.session_state.simulation_running = True
                st.session_state.last_update_time = time.time()
    
    with col2:
        st.session_state.zoom_level = st.slider(
            "Radar Zoom", 
            0.5, 2.0, 1.0, 0.1,
            key='zoom_slider'
        )
    
    # Run simulation step if running - UPDATED SECTION
    if st.session_state.simulation_running:
        current_time = time.time()
        if current_time - st.session_state.last_update_time >= 1.0:
            run_simulation_step()
            st.session_state.last_update_time = current_time
            st.experimental_rerun()  # Only rerun after updating
    
    # Handle button actions
    if execute_btn:
        target_id = st.session_state.selected_target
        if target_id and target_id in st.session_state.targets['id'].values:
            command = st.session_state.get('command_selector', 'Track')
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Command '{command}' issued for {target_id}")
            
            if command == 'Track':
                st.session_state.targets.loc[st.session_state.targets['id'] == target_id, 'status'] = 'Tracking'
            elif command == 'Prioritize':
                st.session_state.targets.loc[st.session_state.targets['id'] == target_id, 'status'] = 'Prioritized'
            elif command == 'Intercept':
                manual_assign_interceptor(target_id, st.session_state.ml_model)
            
            st.session_state.display_mode = 'Selected Target'
    
    if track_all_btn:
        if not st.session_state.targets.empty:
            st.session_state.targets.loc[st.session_state.targets['status'].isin(['Tracking', 'Prioritized']), 'status'] = 'Tracking'
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: All drones set to Tracking")
            st.session_state.display_mode = 'All Drones'
    
    if assign_btn:
        target_id = st.session_state.selected_target
        if target_id and target_id in st.session_state.targets['id'].values:
            manual_assign_interceptor(target_id, st.session_state.ml_model)
            st.session_state.display_mode = 'Selected Target'
    
    if auto_assign_btn:
        assign_interceptors(auto_all=True)
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Automatic assignment triggered for all prioritized drones")
    
    # Update UI state
    update_ui()
    
    # Data Feed
    st.subheader("Tactical Data Feed")
    data_feed_container = st.container()
    with data_feed_container:
        if st.session_state.get('data_feed'):
            for entry in st.session_state.data_feed[:4]:
                st.markdown(f"<div class='data-feed'>{entry}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='data-feed'>No active engagements</div>", unsafe_allow_html=True)
    
    # Plots
    st.subheader("Tactical Display")
    if not st.session_state.targets.empty:
        fig = create_tactical_plots()
        st.pyplot(fig)
    else:
        st.info("No targets detected. Waiting for drone swarm...")
    
    # Command Log
    st.subheader("Command History")
    log_container = st.container()
    with log_container:
        log_html = "<div class='log-panel'>"
        for entry in st.session_state.command_log[-10:]:
            if "ALERT" in entry:
                log_html += f"<div style='color:#ff3333; margin: 5px 0;'>{entry}</div>"
            elif "Error" in entry:
                log_html += f"<div style='color:#ff9900; margin: 5px 0;'>{entry}</div>"
            elif "assigned" in entry:
                log_html += f"<div style='color:#00ffcc; margin: 5px 0;'>{entry}</div>"
            else:
                log_html += f"<div style='color:#ffffff; margin: 5px 0;'>{entry}</div>"
        log_html += "</div>"
        st.markdown(log_html, unsafe_allow_html=True)
    
    # Data Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Targets")
        if not st.session_state.targets.empty:
            st.dataframe(st.session_state.targets[['id', 'range', 'bearing', 'elevation', 'threat_level', 'status']])
        else:
            st.info("No targets detected")
    
    with col2:
        st.subheader("Interceptors")
        if not st.session_state.interceptors.empty:
            st.dataframe(st.session_state.interceptors[['id', 'speed', 'range_capability', 'status', 'target_id']])
        else:
            st.info("No interceptors available")

if __name__ == "__main__":
    main()
