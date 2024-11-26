import csv
import numpy as np
from src.utils.font import to_bin_array, parse_h_file_to_numpy

if __name__ == "__main__":
    # Parse the hex file and convert to binary matrices
    hex_characters = parse_h_file_to_numpy("./input/font.h")
    characters = [to_bin_array(character) for character in hex_characters]

    # Specify the output CSV file
    output_file = "./output/original_characters.csv"

    # Write the matrices to the CSV file
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for character in characters:
            # Write each row of the 7x5 matrix
            writer.writerows(character)
