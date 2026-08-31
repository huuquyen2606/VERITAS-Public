import os
import gc
import json
import time
import shutil
import torch
import requests
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# We assume you still have the utils folder for Logger and ExperimentManager.
# If you don't, you can safely replace them with standard print() statements!
from ...utils import Logger, ExperimentManager

# Import the architecture defined in arch.py
from .arch import create_hex_vocabulary, HexTokenizer, get_malgpt_model

# ==========================================
# 1. CONFIGURATION (100% Faithful to Paper)
# ==========================================

CONFIG = {
    "seed": 42,
    "hex_chunk_size": 4,  # Hex delimited by sets of 4 characters
    "max_length": 1024,  # GPT-2 token context window
    "training_iterations": 1000,  # Paper specifies exactly 1,000 iterations
    "batch_size": 1,  # Small batch size to handle memory overhead
    "max_perturbation_size": 10240,  # 10 KB strict append limit for stealth
}

# ==========================================
# 2. DATASET HELPERS
# ==========================================


class BenignHexDataset(Dataset):
    """PyTorch Dataset to feed benign hex chunks into the GPT-2 trainer."""

    def __init__(self, raw_bytes_list, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.examples = []

        for raw_byte_array in tqdm(raw_bytes_list, desc="Tokenizing benign samples"):
            # Convert Numpy uint8 array to space-delimited hex string
            hex_string = "".join([format(b, "02X") for b in raw_byte_array])
            chunked_hex = " ".join(
                [
                    hex_string[i : i + CONFIG["hex_chunk_size"]]
                    for i in range(0, len(hex_string), CONFIG["hex_chunk_size"])
                ]
            )

            token_ids = tokenizer.encode(
                chunked_hex, max_length=max_length, truncation=True
            )
            if len(token_ids) > 0:
                self.examples.append(torch.tensor(token_ids, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def data_collator(features, pad_token_id):
    """Pads dynamic sequences for the causal language model."""
    max_len = max(len(f["input_ids"]) for f in features)
    batch = {"input_ids": [], "labels": [], "attention_mask": []}

    for f in features:
        input_ids = f["input_ids"]
        seq_len = len(input_ids)
        padding_len = max_len - seq_len

        padded_input = torch.cat(
            [input_ids, torch.full((padding_len,), pad_token_id, dtype=torch.long)]
        )

        attention_mask = torch.cat(
            [
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(padding_len, dtype=torch.long),
            ]
        )

        batch["input_ids"].append(padded_input)
        batch["labels"].append(padded_input.clone())
        batch["attention_mask"].append(attention_mask)

    batch["input_ids"] = torch.stack(batch["input_ids"])
    batch["labels"] = torch.stack(batch["labels"])
    batch["attention_mask"] = torch.stack(batch["attention_mask"])
    return batch


# ==========================================
# 3. MAIN CONTROLLER CLASS (STANDALONE)
# ==========================================


class MalGPTModel:  # <-- REMOVED INHERITANCE
    """
    MalGPT: Single-Shot Black-Box Adversarial Attack Generator.
    Standalone implementation.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.vocab = None

        base_dir = os.path.dirname(__file__)
        self.local_weights_dir = os.path.join(base_dir, "weights")

    def train(self, benign_npz_path, experiment_name=None, save_path=None):
        """
        Trains the GPT-2 model on benign raw bytes to learn safe generation patterns.
        """
        manager = ExperimentManager(
            base_dir=os.path.dirname(__file__), experiment_name=experiment_name
        )
        logger = Logger(manager.run_dir)
        logger.log("[*] Starting MalGPT Training Phase")

        # 1. Setup Architecture & Vocab
        self.vocab = create_hex_vocabulary()
        self.tokenizer = HexTokenizer(self.vocab)
        self.model = get_malgpt_model(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        ).to(self.device)

        # 2. Load Data from .npz and FILTER FOR BENIGN ONLY
        logger.log(f"[*] Loading dataset from {benign_npz_path}")
        data = np.load(benign_npz_path, allow_pickle=True)
        all_labels = data["label"]
        all_raw_bytes = data["raw_byte"]

        # Filter: Keep only samples where label is 0 or "benign"
        benign_raw_bytes_list = []
        for i, lbl in enumerate(all_labels):
            lbl_str = str(lbl).strip().lower()
            if lbl_str == "0" or lbl_str == "benign":
                benign_raw_bytes_list.append(all_raw_bytes[i])

        logger.log(
            f"[*] Filtered {len(benign_raw_bytes_list)} benign samples for training out of {len(all_labels)} total files."
        )

        if len(benign_raw_bytes_list) == 0:
            raise ValueError(
                "No benign samples found in the dataset! Check your labels."
            )

        train_dataset = BenignHexDataset(
            benign_raw_bytes_list, self.tokenizer, CONFIG["max_length"]
        )

        # 3. Configure Trainer
        training_args = TrainingArguments(
            output_dir=manager.run_dir,
            max_steps=CONFIG["training_iterations"],
            per_device_train_batch_size=CONFIG["batch_size"],
            logging_steps=100,
            save_steps=500,
            eval_strategy="no",
            seed=CONFIG["seed"],
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda features: data_collator(
                features, self.tokenizer.pad_token_id
            ),
        )

        # 4. Train & Save
        logger.log("[*] Commencing 1,000 training iterations...")
        trainer.train()

        if save_path:
            model_save_path = save_path
            os.makedirs(model_save_path, exist_ok=True)
        else:
            model_save_path = os.path.join(self.local_weights_dir, "malgpt_weights")

        trainer.save_model(model_save_path)
        with open(os.path.join(model_save_path, "hex_vocab.json"), "w") as f:
            json.dump(self.vocab, f)

        logger.log(f"[*] Training complete. Weights saved to {model_save_path}")

        del trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_weights(self, weights_path=None):
        """Loads the saved custom vocabulary and GPT-2 weights."""
        if weights_path and os.path.exists(weights_path):
            final_path = weights_path
        else:
            final_path = os.path.join(self.local_weights_dir, "malgpt_weights")

        print(f"[*] Loading MalGPT from: {final_path}")

        # Load Vocab
        vocab_path = os.path.join(final_path, "hex_vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}.")

        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)
        self.tokenizer = HexTokenizer(self.vocab)

        # Initialize and Load Model
        self.model = get_malgpt_model(
            vocab_size=self.tokenizer.vocab_size,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Load weights explicitly
        model_file = os.path.join(final_path, "model.safetensors")
        if not os.path.exists(model_file):
            model_file = os.path.join(final_path, "pytorch_model.bin")

        state_dict = torch.load(model_file, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("[*] MalGPT loaded and ready for generation.")

    def _verify_vt_functionality(self, file_path, vt_api_key):
        """Uploads file to VirusTotal to ensure the append attack didn't break functionality."""
        headers = {"x-apikey": vt_api_key}

        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    "https://www.virustotal.com/api/v3/files",
                    headers=headers,
                    files={"file": f},
                )
            if response.status_code != 200:
                return False

            analysis_id = response.json()["data"]["id"]

            start_time = time.time()
            while time.time() - start_time < 300:
                res = requests.get(
                    f"https://www.virustotal.com/api/v3/analyses/{analysis_id}",
                    headers=headers,
                )
                if res.status_code == 200:
                    data = res.json()
                    if data["data"]["attributes"]["status"] == "completed":
                        malicious = data["data"]["attributes"]["stats"].get(
                            "malicious", 0
                        )
                        return malicious > 0
                time.sleep(15)
            return False
        except Exception as e:
            print(f"[!] VT API Error: {e}")
            return False

    def generate_functional_samples(self, target_npz_path, output_dir, vt_api_key):
        """
        End-to-end factory: Ingests raw malware, generates payload, appends, verifies, and saves.
        """
        if not self.model:
            raise RuntimeError("Model weights must be loaded first.")

        os.makedirs(output_dir, exist_ok=True)
        temp_dir = os.path.join(output_dir, "temp_processing")
        os.makedirs(temp_dir, exist_ok=True)

        print(f"[*] Reading target malware from {target_npz_path}")
        data = np.load(target_npz_path, allow_pickle=True)
        all_names = data["name"]
        all_labels = data["label"]
        all_raw_bytes = data["raw_byte"]

        # Filter: Extract ONLY malware (ignore benign/0)
        malware_indices = []
        for i, lbl in enumerate(all_labels):
            lbl_str = str(lbl).strip().lower()
            if lbl_str != "0" and lbl_str != "benign":
                malware_indices.append(i)

        names = [all_names[i] for i in malware_indices]
        labels = [all_labels[i] for i in malware_indices]
        raw_bytes_list = [all_raw_bytes[i] for i in malware_indices]

        print(f"[*] Filtered {len(names)} MALWARE targets for evasion testing.")

        success_count = 0

        for i in tqdm(range(len(names)), desc="Generating & Verifying"):
            hash_name = names[i]
            family_label = labels[i]
            original_bytes = bytes(raw_bytes_list[i])

            hex_string = "".join([format(b, "02X") for b in original_bytes])
            chunked_hex = " ".join(
                [
                    hex_string[i : i + CONFIG["hex_chunk_size"]]
                    for i in range(0, len(hex_string), CONFIG["hex_chunk_size"])
                ]
            )

            inputs = self.tokenizer(
                chunked_hex, max_length=CONFIG["max_length"] - 512, return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0][input_ids.shape[1] :]
            perturbation_hex = self.tokenizer.decode(new_tokens).replace(" ", "")

            clean_hex = "".join(
                c for c in perturbation_hex if c.upper() in "0123456789ABCDEF"
            )
            if len(clean_hex) % 2 != 0:
                clean_hex = clean_hex[:-1]

            perturbation_bytes = bytes.fromhex(clean_hex)
            adv_bytes = (
                original_bytes + perturbation_bytes[: CONFIG["max_perturbation_size"]]
            )

            temp_file = os.path.join(temp_dir, f"{hash_name}.exe")
            with open(temp_file, "wb") as f:
                f.write(adv_bytes)

            is_functional = self._verify_vt_functionality(temp_file, vt_api_key)

            if is_functional:
                final_folder = os.path.join(output_dir, str(family_label))
                os.makedirs(final_folder, exist_ok=True)
                shutil.move(temp_file, os.path.join(final_folder, f"{hash_name}.exe"))
                success_count += 1
                print(f"  [+] Success: {hash_name} (Family: {family_label})")
            else:
                os.remove(temp_file)

            time.sleep(15)

        shutil.rmtree(temp_dir)
        print(
            f"[*] Factory Complete! {success_count}/{len(names)} functional samples saved to {output_dir}"
        )
        return output_dir
