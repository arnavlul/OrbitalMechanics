import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, RangeSlider # Import Widgets

# --- CONFIGURATION ---
SIM_FILE = "hnn_sim.csv"   
ERR_FILE = "hnn_errors.csv"     
REAL_FILE = "./test_files/earth_test.csv"    

def plot_dashboard():
    # 1. Setup the Figure
    fig = plt.figure(figsize=(14, 9)) # Increased height slightly for sliders
    plt.subplots_adjust(bottom=0.25)  # Make room at the bottom for sliders

    # --- Plot 1: Training Loss Breakdown ---
    ax1 = fig.add_subplot(2, 2, 1)
    
    # Global variables to store data limits for sliders
    max_epoch = 100
    max_error = 1000.0

    try:
        df_err = pd.read_csv(ERR_FILE)
        df_err.columns = df_err.columns.str.strip()
        
        required_cols = ['Epoch', 'Total Error', 'MSE Error', 'Energy Error']
        if all(col in df_err.columns for col in required_cols):
            # Store data for update function
            max_epoch = df_err['Epoch'].max()
            max_error = df_err['Total Error'].max()

            # Plot Lines
            l1, = ax1.plot(df_err['Epoch'], df_err['Total Error'], 'purple', label='Total', linewidth=2)
            l2, = ax1.plot(df_err['Epoch'], df_err['MSE Error'], 'blue', linestyle='--', label='MSE', alpha=0.7)
            l3, = ax1.plot(df_err['Epoch'], df_err['Energy Error'], 'green', linestyle=':', label='Energy', alpha=0.9, linewidth=2)
            
            ax1.set_title("Training Loss (Interactive)")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "Column mismatch", ha='center')
    except FileNotFoundError:
        ax1.text(0.5, 0.5, "hnn_errors.csv Not Found", ha='center')

    # --- Plot 2: 3D Orbit ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    try:
        df_sim = pd.read_csv(SIM_FILE)
        ax2.plot(df_sim['x'], df_sim['y'], df_sim['z'], label='AI Prediction', color='blue')
        
        # Load Real Data
        try:
            df_real = pd.read_csv(REAL_FILE, header=None)
            df_real.columns = ['JD', 'x', 'y', 'z', 'vx', 'vy', 'vz']
            limit = min(len(df_real), len(df_sim))
            ax2.plot(df_real['x'][:limit], df_real['y'][:limit], df_real['z'][:limit], 
                     color='red', linestyle='--', alpha=0.5, label='Real Physics')
        except:
            pass
            
    except FileNotFoundError:
        pass

    ax2.set_title("3D Orbit Trajectory")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend()

    # --- Plot 3: 2D Top-Down View ---
    ax3 = fig.add_subplot(2, 2, 3)
    try:
        ax3.plot(df_sim['x'], df_sim['y'], label='AI', color='blue')
        if 'df_real' in locals():
            ax3.plot(df_real['x'][:limit], df_real['y'][:limit], 'r--', alpha=0.5, label='Real')
    except:
        pass
        
    ax3.set_title("Top-Down View (X-Y)")
    ax3.axis('equal')
    ax3.grid(True)

    # --- INTERACTIVITY SECTION ---
    
    # 1. Define Axes for Sliders [left, bottom, width, height]
    ax_epoch = plt.axes([0.15, 0.1, 0.35, 0.03])
    ax_ymax  = plt.axes([0.60, 0.1, 0.30, 0.03])

    # 2. Create Sliders
    # RangeSlider for X-Axis (Epochs)
    slider_epoch = RangeSlider(ax_epoch, "Epoch Range", 1, max_epoch, valinit=(1, max_epoch))
    
    # Standard Slider for Y-Axis Max (Clipping high values)
    slider_ymax = Slider(ax_ymax, "Max Loss (Y)", 0.1, max_error, valinit=max_error)

    # 3. Update Function
    def update(val):
        # Update X Limits (Epochs)
        min_e, max_e = slider_epoch.val
        ax1.set_xlim(min_e, max_e)
        
        # Update Y Limits (Loss Cutoff)
        # We keep Y-min at 0 because loss implies magnitude
        ax1.set_ylim(0, slider_ymax.val)
        
        # Redraw
        fig.canvas.draw_idle()

    # 4. Connect Sliders
    slider_epoch.on_changed(update)
    slider_ymax.on_changed(update)

    # IMPORTANT: Keep references to widgets so they aren't garbage collected
    fig.sliders = [slider_epoch, slider_ymax]

    plt.show()

if __name__ == "__main__":
    plot_dashboard()