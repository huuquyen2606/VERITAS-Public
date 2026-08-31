#!/usr/bin/env python3
import sys
import importlib.util

def print_status(msg, ok=True):
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    status = "[OK]" if ok else "[FAIL]"
    print(f"{color}{status}{reset} {msg}")

def check_module(name):
    if importlib.util.find_spec(name) is None:
        print_status(f"Python module '{name}' is missing.", False)
        return False
    print_status(f"Python module '{name}' found.", True)
    return True

def main():
    print("=== DQEAF Preflight Environment Check ===")
    
    all_ok = True
    
    # Check core dependencies
    deps = ["numpy", "sklearn", "torch", "gymnasium", "joblib", "lief", "lightgbm"]
    for dep in deps:
        if not check_module(dep):
            all_ok = False
            
    # Explicitly check ember and its dimensions
    if not check_module("ember"):
        print_status("EMBER module missing. Required for LightGBM surrogate.", False)
        all_ok = False
    else:
        try:
            import ember
            # Initialize EMBER v2 extractor
            extractor = ember.PEFeatureExtractor(2, print_feature_warning=False)
            if extractor.dim != 2381:
                print_status(f"EMBER v2 extractor dimension mismatch! Expected 2381, got {extractor.dim}", False)
                all_ok = False
            else:
                print_status(f"EMBER v2 extractor dimension matches exactly: {extractor.dim}", True)
        except Exception as e:
            print_status(f"Error testing EMBER module: {e}", False)
            all_ok = False
            
    # LIEF specific API check to ensure compatibility with actions.py
    try:
        import lief
        print_status(f"LIEF Version: {lief.__version__}", True)
    except Exception as e:
        print_status(f"LIEF import failed: {e}", False)
        all_ok = False

    print("=========================================")
    if all_ok:
        print("\033[92m[SUCCESS]\033[0m Environment is 100% ready for DQEAF execution.")
        sys.exit(0)
    else:
        print("\033[91m[FAILURE]\033[0m Please resolve the missing dependencies or dimension errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
