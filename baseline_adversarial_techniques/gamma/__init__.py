"""
GAMMA Attack Implementation
===========================

Based on the paper:
Demetrio, Luca, et al. "Functionality-preserving black-box optimization of adversarial windows malware."
IEEE Transactions on Information Forensics and Security (2021).
"""

import os
import random
import traceback
from secml.array import CArray
from secml_malware.attack.blackbox.c_gamma_sections_evasion import (
    CGammaSectionsEvasionProblem,
)
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from . import arch
from .utils import data_utils

# ==========================================
# 1. GAMMA CONFIGURATION 
# ==========================================
CFG = {
    "POPULATION_SIZE": 10,  # Number of variants per generation
    "ITERATIONS": 51,       # 51 iterations * 10 pop = ~510 queries max
    "REGULARIZATION": 1e-4, # Lambda penalty for payload size
    "N_SECTIONS": 75,       # Number of benign sections to extract
    "SECTION_TYPES": [".rdata"],  # Target sections for injection
    "SEED": 42,             # Seed for reproducibility
}

# ==========================================
# 2. GENERATION FUNCTION
# ==========================================
def create_adversarial_malware(
    malware_path,
    section_population,  
    output_dir,
    custom_model,
    original_label,
    config=CFG,  
):
    """
    Generates a single optimal adversarial malware using GAMMA.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load Malware
    if not os.path.isfile(malware_path):
        raise FileNotFoundError(f"Malware file not found: {malware_path}")

    x_init = data_utils.load_malware_to_carray(malware_path)
    filename = os.path.basename(malware_path)

    # 2. Initialize Model Wrapper
    model_wrapper = arch.MyCustomModelWrapper(custom_model)
    model_wrapper.current_original_label = original_label
    
    # --- CRITICAL FIX #1: Target label set to 0 (Benign) ---
    y_target = CArray([0]) 

    # 3. Define Attack Problem
    current_seed = config.get("SEED", random.randint(0, 999999))

    try:
        # --- CRITICAL FIX #2: Pass section_population and model_wrapper as POSITIONAL arguments ---
        problem = CGammaSectionsEvasionProblem(
            section_population,
            model_wrapper,           
            population_size=config["POPULATION_SIZE"],
            penalty_regularizer=config["REGULARIZATION"],
            iterations=config["ITERATIONS"],
            seed=current_seed,
            hard_label=False,
        )
    except Exception as e:
        print(f"     [ERR] Failed to initialize GAMMA problem: {e}")
        traceback.print_exc()
        return output_dir

    # 4. Initialize Genetic Engine & Run
    try:
        # Pass the problem positionally here as well just to be safe
        genetic_engine = CGeneticAlgorithm(problem)
        
        _, _, adv_ds, _ = genetic_engine.run(x_init, y_target)

        # 5. Extract and Save
        x_adv = adv_ds.X[0, :]
        name_no_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1]

        save_name = f"{name_no_ext}_adv{ext}"
        save_path = os.path.join(output_dir, save_name)

        data_utils.save_carray_to_pe(x_adv, save_path)
        print(f"     [OK] Saved to: {save_path}")

    except Exception as e:
        print(f"     [ERR] Failed to generate variant: {e}")
        traceback.print_exc()

    return output_dir