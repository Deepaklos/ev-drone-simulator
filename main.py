#!/usr/bin/env python3
"""
EV Drone Flight Simulator - Main Entry Point

Launches the GUI and initializes the simulation engine.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from gui.main_window import SimulatorWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt


def main():
    """
    Initialize and launch the simulator GUI.
    """
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("EV Drone Flight Simulator")
    app.setApplicationVersion("0.1.0")
    
    # Create and show main window
    window = SimulatorWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
