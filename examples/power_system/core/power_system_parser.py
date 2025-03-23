import pypsa
import configparser
import os
import pandas as pd

class PowerSystemParser:
    """Parser for INI files to create PyPSA networks."""
    
    def __init__(self):
        self.network = pypsa.Network()
        self.config = configparser.ConfigParser()
        
    def parse_ini_file(self, ini_file_path):
        """Parse the INI file and create a PyPSA network."""
        if not os.path.exists(ini_file_path):
            raise FileNotFoundError(f"INI file not found: {ini_file_path}")
            
        # Read the INI file
        self.config.read(ini_file_path)
        
        # Process general settings
        if 'General' in self.config:
            self._process_general_settings()
            
        # Process buses
        if 'Buses' in self.config:
            self._process_buses()
            
        # Process generators
        if 'Generators' in self.config:
            self._process_generators()
            
        # Process loads
        if 'Loads' in self.config:
            self._process_loads()
            
        # Process lines
        if 'Lines' in self.config:
            self._process_lines()
            
        # Process transformers
        if 'Transformers' in self.config:
            self._process_transformers()
            
        # Process storage units
        if 'StorageUnits' in self.config:
            self._process_storage_units()
            
        return self.network
    
    def _process_general_settings(self):
        """Process general settings from the INI file."""
        general = self.config['General']
        
        if 'name' in general:
            self.network.name = general['name']
            
        if 'frequency' in general:
            self.network.frequency = float(general['frequency'])
            
        if 'snapshot_start' in general and 'snapshot_end' in general:
            self.network.set_snapshots(
                pd.date_range(
                    general['snapshot_start'],
                    general['snapshot_end'],
                    freq=general.get('snapshot_freq', 'H')
                )
            )
    
    def _process_buses(self):
        """Process buses from the INI file."""
        for bus_id, bus_config in self.config['Buses'].items():
            # Skip the DEFAULT section
            if bus_id == 'DEFAULT':
                continue
                
            bus_params = self._parse_component_params(bus_config)
            
            # Add required x and y coordinates if missing
            if 'x' not in bus_params:
                bus_params['x'] = 0
            if 'y' not in bus_params:
                bus_params['y'] = 0
                
            self.network.add("Bus", bus_id, **bus_params)
    
    def _process_generators(self):
        """Process generators from the INI file."""
        if 'Generators' not in self.config:
            return
            
        for gen_id, gen_config in self.config['Generators'].items():
            # Skip the DEFAULT section
            if gen_id == 'DEFAULT':
                continue
                
            gen_params = self._parse_component_params(gen_config)
            
            # Bus is required
            if 'bus' not in gen_params:
                raise ValueError(f"Generator {gen_id} missing required 'bus' parameter")
                
            self.network.add("Generator", gen_id, **gen_params)
    
    def _process_loads(self):
        """Process loads from the INI file."""
        if 'Loads' not in self.config:
            return
            
        for load_id, load_config in self.config['Loads'].items():
            # Skip the DEFAULT section
            if load_id == 'DEFAULT':
                continue
                
            load_params = self._parse_component_params(load_config)
            
            # Bus is required
            if 'bus' not in load_params:
                raise ValueError(f"Load {load_id} missing required 'bus' parameter")
                
            self.network.add("Load", load_id, **load_params)
    
    def _process_lines(self):
        """Process lines from the INI file."""
        if 'Lines' not in self.config:
            return
            
        for line_id, line_config in self.config['Lines'].items():
            # Skip the DEFAULT section
            if line_id == 'DEFAULT':
                continue
                
            line_params = self._parse_component_params(line_config)
            
            # Bus1 and bus2 are required
            if 'bus0' not in line_params:
                raise ValueError(f"Line {line_id} missing required 'bus0' parameter")
            if 'bus1' not in line_params:
                raise ValueError(f"Line {line_id} missing required 'bus1' parameter")
                
            self.network.add("Line", line_id, **line_params)
    
    def _process_transformers(self):
        """Process transformers from the INI file."""
        if 'Transformers' not in self.config:
            return
            
        for transformer_id, transformer_config in self.config['Transformers'].items():
            # Skip the DEFAULT section
            if transformer_id == 'DEFAULT':
                continue
                
            transformer_params = self._parse_component_params(transformer_config)
            
            # Bus0 and bus1 are required
            if 'bus0' not in transformer_params:
                raise ValueError(f"Transformer {transformer_id} missing required 'bus0' parameter")
            if 'bus1' not in transformer_params:
                raise ValueError(f"Transformer {transformer_id} missing required 'bus1' parameter")
                
            self.network.add("Transformer", transformer_id, **transformer_params)
    
    def _process_storage_units(self):
        """Process storage units from the INI file."""
        if 'StorageUnits' not in self.config:
            return
            
        for storage_id, storage_config in self.config['StorageUnits'].items():
            # Skip the DEFAULT section
            if storage_id == 'DEFAULT':
                continue
                
            storage_params = self._parse_component_params(storage_config)
            
            # Bus is required
            if 'bus' not in storage_params:
                raise ValueError(f"Storage unit {storage_id} missing required 'bus' parameter")
                
            self.network.add("StorageUnit", storage_id, **storage_params)
    
    def _parse_component_params(self, config_str):
        """Parse component parameters from a config string."""
        params = {}
        for param in config_str.split(','):
            if '=' in param:
                key, value = param.strip().split('=', 1)
                # Try to convert to appropriate type
                try:
                    # Try as float first
                    params[key] = float(value)
                except ValueError:
                    # If not a float, keep as string
                    params[key] = value
                    
        return params
