import os
import numpy as np
import pefile
from capstone import Cs, CS_ARCH_X86, CS_MODE_32, CS_MODE_64
from tqdm import tqdm


# Thực thi đoạn code
TARGET_DIRECTORY = "malguise"
OUTPUT_FILE = "malguise_static.npz"


def extract_all_features(target_dir, output_filepath):
    data_dict = {
        "name": [],
        "label": [],
        "raw_byte": [],
        "op_code": [],
        "api_pefile": [],
    }

    # Cấu hình Capstone disassembler cho x86 (32-bit) và x86_64 (64-bit)
    md_32 = Cs(CS_ARCH_X86, CS_MODE_32)
    md_64 = Cs(CS_ARCH_X86, CS_MODE_64)
    md_32.skipdata = True
    md_64.skipdata = True

    # Duyệt qua các thư mục con (Benign, Locker, Mediyes,...)
    for label in os.listdir(target_dir):
        label_dir = os.path.join(target_dir, label)

        if not os.path.isdir(label_dir):
            continue

        print(f"\n[*] Đang xử lý nhãn (label): {label}")

        for filename in tqdm(os.listdir(label_dir)):
            filepath = os.path.join(label_dir, filename)
            if not os.path.isfile(filepath):
                continue

            # ==========================================
            # 1. TRÍCH XUẤT RAW BYTE
            # ==========================================
            try:
                with open(filepath, "rb") as f:
                    raw_data = f.read()
                raw_byte_arr = np.frombuffer(raw_data, dtype=np.uint8)
            except Exception as e:
                print(f"[!] Lỗi đọc file {filename}: {e}")
                continue

            # ==========================================
            # 2. TRÍCH XUẤT OP CODE & STATIC API
            # ==========================================
            opcodes = []
            apis = []

            try:
                pe = pefile.PE(filepath)

                # --- A. Lấy OP Code ---
                md = md_64 if pe.FILE_HEADER.Machine == 0x8664 else md_32
                for section in pe.sections:
                    if section.Characteristics & 0x20000000:  # Executable
                        code = section.get_data()
                        for instruction in md.disasm(code, section.VirtualAddress):
                            opcodes.append(instruction.mnemonic)

                # --- B. Lấy Static API ---
                # Kiểm tra xem file PE có bảng Import không
                if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in entry.imports:
                            # Tên API có thể bị null hoặc chứa ký tự rác
                            if imp.name is not None:
                                apis.append(imp.name.decode("utf-8", errors="ignore"))

            except Exception:
                # Bỏ qua nếu file PE bị hỏng header hoặc mã hóa
                pass

            # Ghép list thành chuỗi cách nhau bởi khoảng trắng
            opcode_str = " ".join(opcodes)
            api_str = " ".join(apis)

            # ==========================================
            # 3. LƯU VÀO DICTIONARY
            # ==========================================
            data_dict["name"].append(filename)
            data_dict["label"].append(label)
            data_dict["raw_byte"].append(raw_byte_arr)
            data_dict["op_code"].append(opcode_str)
            data_dict["api_pefile"].append(api_str)

    # ==========================================
    # 4. NÉN VÀ XUẤT RA FILE .NPZ
    # ==========================================
    print(f"\n[*] Đang lưu dữ liệu vào {output_filepath}...")
    np.savez_compressed(
        output_filepath,
        name=np.array(data_dict["name"]),
        label=np.array(data_dict["label"]),
        raw_byte=np.array(data_dict["raw_byte"], dtype=object),
        op_code=np.array(data_dict["op_code"], dtype=object),
        api_pefile=np.array(data_dict["api_pefile"], dtype=object),
    )
    print(f"[+] Hoàn tất! Đã lưu thành công tại {output_filepath}")


extract_all_features(TARGET_DIRECTORY, OUTPUT_FILE)
