import numpy as np
from secml.array import CArray

# We still need the genetic algorithm class for the static save method
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm


def load_malware_to_carray(file_path):
    """
    Reads a PE file from disk and converts it to a CArray
    compatible with the genetic engine.
    """
    with open(file_path, "rb") as f:
        code = bytearray(f.read())

    # Convert bytearray to numpy array of uint8
    numpy_data = np.array([x for x in code], dtype=np.uint8)

    # Wrap in CArray and ensure it is 2D (1 sample, N features)
    carray_data = CArray(numpy_data).atleast_2d()
    return carray_data


def save_carray_to_pe(x_adv, output_path):
    """
    Saves the adversarial CArray back to a PE file on disk.
    Wrapper around CGeneticAlgorithm.write_adv_to_file
    """
    # Using the static method provided by the author's code
    CGeneticAlgorithm.write_adv_to_file(x_adv, output_path)
