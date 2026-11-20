import os
import subprocess
import tempfile
import shutil
import csv
import logging
import hashlib
from detector.interface import DetectorInterface, DetectorResult

logger = logging.getLogger(__name__)

class MagicAdapter(DetectorInterface):
    def __init__(self, magic_dir: str, use_probability: bool = True):
        super().__init__(use_probability=use_probability)
        self.magic_dir = os.path.abspath(magic_dir)
        self.maldefender_dir = os.path.join(self.magic_dir, "maldefender")
        self.msacfg_dir = os.path.abspath(os.path.join(self.magic_dir, "..", "MSACFG"))
        
        # We need python path to the virtualenv if any, or just system python
        self.python_bin = os.path.join(self.magic_dir, "..", "det_venv", "bin", "python3")
        if not os.path.exists(self.python_bin):
            self.python_bin = "python3"

    def predict(self, pe_bytes: bytes, preserve_artifacts: bool = False) -> DetectorResult:
        """
        Evaluates a single PE file by:
        1. Saving to temp dir
        2. Running MAGIC's prediction pipeline
        3. Extracting the probability from submission.csv
        """
        cfg_source_sha256 = hashlib.sha256(pe_bytes).hexdigest()
        tmpdir = tempfile.mkdtemp()
        
        try:
            sample_path = os.path.join(tmpdir, "sample.exe")
            with open(sample_path, "wb") as f:
                f.write(pe_bytes)
                
            test_dir = os.path.join(tmpdir, "TestSet")
            os.makedirs(test_dir, exist_ok=True)
            
            # Automatically find the latest checkpoint in msacfg_models
            models_dir = os.path.join(self.maldefender_dir, "msacfg_models")
            import glob
            pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
            if not pt_files:
                raise RuntimeError(f"No .pt models found in {models_dir}. Please train MAGIC first.")
            # Sort by modification time to get the latest
            latest_model_path = max(pt_files, key=os.path.getmtime)
            test_checkpoint = os.path.basename(latest_model_path)
            logger.info(f"Using latest MAGIC checkpoint: {test_checkpoint}")
            
            # Step 1: EXE -> ASM (using prepare_data logic)
            logger.info("Converting EXE to ASM (angr)...")
            prepare_data_py = os.path.join(self.msacfg_dir, "prepare_data.py")
            asm_path = os.path.join(test_dir, "sample.asm")
            
            subprocess.run(
                [self.python_bin, "-c", 
                 f"import sys; sys.path.append('{self.msacfg_dir}'); "
                 f"from prepare_data import exe_to_ida_asm_angr; "
                 f"exe_to_ida_asm_angr('{sample_path}', '{asm_path}')"],
                check=True
            )
            
            # Step 2: ASM -> ACFG (using acfg_pipeline logic)
            logger.info("Converting ASM to ACFG...")
            subprocess.run(
                [self.python_bin, "-c",
                 f"import sys; sys.path.append('{self.maldefender_dir}'); "
                 f"from acfg_pipeline import AcfgMaster; "
                 f"master = AcfgMaster('{test_dir}', outputTxtName='MSACFG_test', binaryIds=['sample']); "
                 f"master.dispatchWorkers(1)"],
                cwd=self.maldefender_dir,
                check=True
            )
            
            # Step 3: ACFG -> Model Inference
            logger.info(f"Running inference using checkpoint {test_checkpoint}...")
            
            # We need to run predict_model.py. 
            # We must trick it into thinking `test_dir` is the TEST_DIR
            subprocess.run(
                [self.python_bin, "predict_model.py",
                 "-seed", "1",
                 "-data", "MSACFG",
                 "-train_dir", test_dir, # Doesn't matter for test
                 "-test_dir", test_dir,
                 "-mode", "cpu",
                 "-gpu_id", "0",
                 "-use_cached_data", "False", # We want to compute fresh features
                 "-cache_path", "cached_msacfg_graphs",
                 "-norm_path", "norm_msacfg",
                 "-norm_op", "min_max",
                 "-model_date", test_checkpoint.split('_')[0],
                 "-hp_path", "custom.hp"],
                cwd=self.maldefender_dir,
                check=True
            )
            
            # Step 4: Parse probability
            sub_csv = os.path.join(test_dir, "submission.csv")
            with open(sub_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    prob = float(row['MalwareProb'])
                    is_malware = prob >= 0.5
                    
                    # MAGIC does not output JSON CFGs, it outputs an angr ASM file
                    # which is then converted to txt/npz. We will preserve the ASM file as the CFG.
                    cfg_path = asm_path if os.path.exists(asm_path) else None
                    
                    return DetectorResult(
                        malware_prob=prob, 
                        is_malware=is_malware,
                        cfg_path=cfg_path,
                        cfg_source_sha256=cfg_source_sha256
                    )
            
            raise RuntimeError("Could not find probability in submission.csv")
            
        finally:
            if not preserve_artifacts:
                shutil.rmtree(tmpdir, ignore_errors=True)

