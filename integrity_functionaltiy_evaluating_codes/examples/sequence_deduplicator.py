import time

def deduplicate_sequence(api_sequence, lm=5, k=2):
    """
    Nén các vòng lặp API dư thừa từ Dynamic Analysis.
    
    Tham số:
    - api_sequence (list): Danh sách các API calls (có thể là string hoặc dict).
    - lm (int): Độ dài tối đa của một chuỗi pattern cần tìm kiếm (Maximum pattern length).
    - k (int): Số lần lặp liên tiếp tối đa được phép giữ lại (Max consecutive duplicates).
    
    Trả về:
    - list: Mảng API sau khi đã được nén nhiễu.
    """
    if not api_sequence:
        return []
    
    # Tạo một bản sao để thao tác
    result = list(api_sequence)
    
    # Lặp lại quá trình nén cho đến khi độ dài mảng không còn thay đổi
    while True:
        prev_len = len(result)
        
        # Quét các pattern có độ dài từ 1 đến lm (Theo nghiên cứu: tối đa 5)
        for p in range(1, lm + 1):
            i = 0
            new_seq = []
            
            while i < len(result):
                pattern = result[i : i + p]
                
                # Nếu phần còn lại của mảng không đủ dài để chứa 1 pattern lặp lại, 
                # đẩy toàn bộ phần còn lại vào new_seq và kết thúc vòng lặp hiện tại.
                if i + 2 * p > len(result):
                    new_seq.extend(result[i:])
                    break
                    
                # Đếm số lần pattern lặp lại liên tiếp
                repeats = 1
                idx = i + p
                while idx + p <= len(result) and result[idx : idx + p] == pattern:
                    repeats += 1
                    idx += p
                    
                # Áp dụng logic cắt tỉa: Nếu lặp > k lần, chỉ giữ lại đúng k lần
                if repeats > k:
                    for _ in range(k):
                        new_seq.extend(pattern)
                    # Bước nhảy chỉ mục: Bỏ qua toàn bộ phần nhiễu đã đếm
                    i = idx 
                else:
                    # Nếu không vượt ngưỡng k, chỉ ghi nhận phần tử hiện tại và nhích lên 1 bước
                    new_seq.append(result[i])
                    i += 1
                    
            result = new_seq
            
        # Nếu sau khi quét từ p=1 đến p=lm mà độ dài mảng không đổi -> Đã nén tối đa
        if len(result) == prev_len:
            break
            
    return result


# ==========================================
# KHỐI TEST (Chỉ chạy khi thực thi trực tiếp file này)
# ==========================================
if __name__ == "__main__":
    # Mô phỏng một chuỗi API bị nhiễu do vòng lặp
    raw_dynamic_trace = [
        "LoadLibraryA",
        "GetProcAddress",
        # Mô phỏng vòng lặp mạng (Pattern dài 2 lặp lại 6 lần)
        "InternetOpenA", "InternetConnectA",
        "InternetOpenA", "InternetConnectA",
        "InternetOpenA", "InternetConnectA",
        "InternetOpenA", "InternetConnectA",
        "InternetOpenA", "InternetConnectA",
        "InternetOpenA", "InternetConnectA",
        # Mô phỏng API rác (Pattern dài 1 lặp lại 5 lần)
        "Sleep", "Sleep", "Sleep", "Sleep", "Sleep",
        "CreateFileA",
        "WriteFile"
    ]
    
    print(f"[*] Độ dài chuỗi gốc: {len(raw_dynamic_trace)}")
    print("[*] Chuỗi gốc:", raw_dynamic_trace)
    
    start_time = time.time()
    # Chạy nén với cấu hình chuẩn: Pattern max 5, lặp max 2
    compressed_trace = deduplicate_sequence(raw_dynamic_trace, lm=5, k=2)
    end_time = time.time()
    
    print(f"\n[*] Độ dài sau nén : {len(compressed_trace)}")
    print("[*] Chuỗi sau nén:", compressed_trace)
    print(f"[*] Thời gian xử lý: {(end_time - start_time)*1000:.4f} ms")