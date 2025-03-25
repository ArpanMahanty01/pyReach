
import sys
from power_system.core.power_system_visualizer import PowerSystemVisualizer

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ini_file_path = sys.argv[1]
    else:
        raise ValueError("Please provide the path to the INI file.")
    
    app = PowerSystemVisualizer()
    app.file_path_var.set(ini_file_path)
    app.load_network()
    app.run()