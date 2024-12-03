import yaml


def save_config(config, outputs_dir):
    """
    Saves a config dictionary as both a pickle file and a YAML file, ensuring only basic types are saved.
    Also, lists like 'mean' and 'std' are saved in flow style (on a single line).
    
    Args:
        config (dict): The configuration dictionary to save.
        outputs_dir (str): The directory path where the files will be saved.
    """

    def is_basic_type(value):
        """
        Check if a value is a basic data type that can be saved in YAML.
        Basic types include int, float, str, bool, list, and dict.
        """
        return isinstance(value, (int, float, str, bool, list, dict, type(None)))

    def filter_dict(data_dict):
        """
        Recursively filter out any keys from the dictionary whose values contain non-basic types (e.g., objects).
        """
        if not isinstance(data_dict, dict):
            return data_dict
        
        filtered_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Recursively filter nested dictionaries
                nested_dict = filter_dict(value)
                if nested_dict:  # Only add non-empty dictionaries
                    filtered_dict[key] = nested_dict
            elif is_basic_type(value):
                # Add if the value is a basic type
                filtered_dict[key] = value
            else:
                # Skip the key if the value is not a basic type (e.g., an object)
                print(f"Skipping key '{key}' because its value is of type {type(value)}")
        
        return filtered_dict

    def save_dict_to_yaml(data_dict, file_path):
        """
        Saves a dictionary to a YAML file, excluding any keys where the value is an object or contains an object.
        Additionally, ensures that specific lists (like 'mean' and 'std') are saved in flow style.
        
        Args:
            data_dict (dict): The dictionary to save.
            file_path (str): The local file path where the YAML file will be saved.
        """
        
        # Custom representer for lists to force flow style (compact lists)
        class FlowStyleList(list):
            pass
        
        def flow_style_list_representer(dumper, data):
            return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        
        yaml.add_representer(FlowStyleList, flow_style_list_representer)

        # Preprocess specific lists to be in flow style
        if 'mean' in data_dict:
            data_dict['mean'] = FlowStyleList(data_dict['mean'])
        if 'std' in data_dict:
            data_dict['std'] = FlowStyleList(data_dict['std'])

        try:
            # Filter the dictionary
            filtered_dict = filter_dict(data_dict)
            
            # Save the filtered dictionary as YAML
            with open(file_path, 'w') as f:
                yaml.dump(filtered_dict, f, default_flow_style=False)  # Save with default block style except for FlowStyleList
            print(f"Filtered dictionary successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving dictionary to YAML: {e}")

    # Save as YAML
    save_dict_to_yaml(config, outputs_dir + '/config.yaml')