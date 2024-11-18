import re
import numpy as np

def parse_h_file_to_numpy(h_file):
    with open(h_file, 'r') as file:
        content = file.read()
    
    # Extract the array using a regular expression
    matches = re.search(r'Font3\[.*?] = {(.*?)};', content, re.DOTALL)
    if not matches:
        raise ValueError("Could not find the Font3 array in the file.")
    
    array_content = matches.group(1)
    # Remove C-style comments (// and everything after it on the same line)
    array_content = re.sub(r'//.*', '', array_content)
    # Replace curly braces with square brackets
    array_content = array_content.replace('{', '[').replace('}', ']')
    # Remove trailing commas that may remain after comment removal
    array_content = re.sub(r',\s*]', ']', array_content)
    # Remove unnecessary spaces and newlines
    array_content = re.sub(r'\s+', ' ', array_content)
    
    # Evaluate the cleaned-up string as a Python list
    python_list = eval(array_content)
    
    # Convert to a NumPy array
    numpy_array = np.array(python_list, dtype=np.uint8)  # Use uint8 for hexadecimal data
    
    return numpy_array

def to_bin_array(encoded_caracter):
    bin_array = np.zeros((7, 5), dtype=int)
    for row in range(0, 7):
        current_row = encoded_caracter[row]
        for col in range(0, 5):
            bin_array[row][4-col] = current_row & 1
            current_row >>= 1
    return bin_array