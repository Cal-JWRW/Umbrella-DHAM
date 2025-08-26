import numpy as np

def read_log(filepath):
    """
    Reads a log file from an umbrella sampling simulation and extracts:
    - ensemble temperature
    - umbrella center (init)
    - biasing constant (k)
    
    Parameters:
        filepath (str): Path to the log file.
    
    Returns:
        temp (float): Ensemble temperature.
        uc (float): Umbrella center value.
        k (float): Biasing constant.
    
    Raises:
        ValueError: If any of the required values cannot be found in the log.
    """

    # Initialize variables with default "not found" values
    temp = -1      # Temperature, will remain -1 if not found
    uc = None      # Umbrella center, will remain None if not found
    k = -1         # Biasing constant, will remain -1 if not found

    # Open the log file for reading
    with open(filepath, 'r') as file:
        # Loop over each line in the file
        for l in file:
            sl = l.split('=')  # Split line into key and value at '='
            
            # Remove leading/trailing whitespace and check the key
            if sl[0].strip() == 'ensemble-temperature':
                temp = float(sl[1].strip())  # Convert value to float
            elif sl[0].strip() == 'init':
                uc = float(sl[1].strip())    # Convert value to float
            elif sl[0].strip() == 'k':
                k = float(sl[1].strip())     # Convert value to float

    # Check if temperature was found, raise error if not
    if temp == -1:
        raise ValueError(f"Temperature cannot be found in log {filepath}, please check for malformed logfile!")

    # Check if umbrella center or biasing constant were found, raise error if not
    if uc is None or k == -1:
        raise ValueError(f"Umbrella center / Biasing Constant cannot be found in log {filepath}, please check this is an umbrella sampling logfile!")

    # Return the extracted values
    return temp, uc, k