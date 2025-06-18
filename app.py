import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap

# Initialize session state
def init_session_state():
    # Initialize all state variables with default values
    defaults = {
        'targets': pd.DataFrame({
            'id': ['T1'],
            'range': [180.0],
            'bearing': [45.0],
            'elevation': [5.0],
            'velocity': [-0.03],
            'threat_level': [0.7],
            'status': ['Tracking'],
            'x': [180.0 * np.cos(np.radians(45.0))],
            'y': [180.0 * np.sin(np.radians(45.0))]
        }),
        'interceptors': pd.DataFrame({
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
        }),
        'command_log': [],
        'last_spawn_time': time.time(),
        'sweep_angle': 0,
        'neutralization_flash': {},
        'display_mode': 'All Drones',
        'plot_axes': {},
        'simulation_running': False,
        'last_update_time': time.time(),
        'rerun_requested': False,
        'selected_target': 'T1',
        'ml_model': 'Neural Network',
        'zoom_level': 1.0,
        'data_feed': []
    }
    
    # Set defaults for any missing keys
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# [Rest of the functions remain the same as in the previous solution]

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="Quantum Radar Mission Simulator",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state first thing
    init_session_state()
    
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
    
    # Header
    st.title("🚀 Quantum Radar Mission Simulator")
    st.markdown("---")
    
    # Control Panel
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.session_state.selected_target = st.selectbox(
                "Select Target", 
                st.session_state.targets['id'].tolist(),
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
    
    # Simulation Control
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.session_state.simulation_running:
            if st.button("⏹️ Stop Simulation"):
                st.session_state.simulation_running = False
                st.session_state.rerun_requested = True
        else:
            if st.button("▶️ Start Simulation"):
                st.session_state.simulation_running = True
                st.session_state.last_update_time = time.time()
                st.session_state.rerun_requested = True
    
    with col2:
        st.session_state.zoom_level = st.slider(
            "Radar Zoom", 
            0.5, 2.0, 1.0, 0.1,
            key='zoom_slider'
        )
    
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
            st.session_state.rerun_requested = True
    
    if track_all_btn:
        if not st.session_state.targets.empty:
            st.session_state.targets.loc[st.session_state.targets['status'].isin(['Tracking', 'Prioritized']), 'status'] = 'Tracking'
            st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: All drones set to Tracking")
            st.session_state.display_mode = 'All Drones'
            st.session_state.rerun_requested = True
    
    if assign_btn:
        target_id = st.session_state.selected_target
        if target_id and target_id in st.session_state.targets['id'].values:
            manual_assign_interceptor(target_id, st.session_state.ml_model)
            st.session_state.display_mode = 'Selected Target'
            st.session_state.rerun_requested = True
    
    if auto_assign_btn:
        assign_interceptors(auto_all=True)
        st.session_state.command_log.append(f"{time.strftime('%H:%M:%S')}: Automatic assignment triggered for all prioritized drones")
        st.session_state.rerun_requested = True
    
    # Run simulation step if running
    if st.session_state.simulation_running:
        current_time = time.time()
        if current_time - st.session_state.last_update_time >= 1.0:
            run_simulation_step()
            st.session_state.last_update_time = current_time
            st.session_state.rerun_requested = True
    
    # Update UI state
    update_ui()
    
    # Data Feed
    st.subheader("📡 Tactical Data Feed")
    data_feed_container = st.container()
    with data_feed_container:
        if st.session_state.data_feed:
            for entry in st.session_state.data_feed[:4]:
                st.markdown(f"<div class='data-feed'>{entry}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='data-feed'>No active engagements</div>", unsafe_allow_html=True)
    
    # Plots
    st.subheader("📊 Tactical Display")
    if not st.session_state.targets.empty:
        fig = create_tactical_plots()
        st.pyplot(fig)
    else:
        st.info("No targets detected. Waiting for drone swarm...")
    
    # Command Log
    st.subheader("📝 Command History")
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
        st.subheader("🎯 Targets")
        if not st.session_state.targets.empty:
            st.dataframe(st.session_state.targets[['id', 'range', 'bearing', 'elevation', 'threat_level', 'status']])
        else:
            st.info("No targets detected")
    
    with col2:
        st.subheader("🛡️ Interceptors")
        if not st.session_state.interceptors.empty:
            st.dataframe(st.session_state.interceptors[['id', 'speed', 'range_capability', 'status', 'target_id']])
        else:
            st.info("No interceptors available")
            
    # Handle rerun requests at the end of the script
    if st.session_state.rerun_requested:
        st.session_state.rerun_requested = False
        st.experimental_rerun()

if __name__ == "__main__":
    main()
