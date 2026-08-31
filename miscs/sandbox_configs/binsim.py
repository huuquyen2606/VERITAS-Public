import os
import shutil
import json
import time
import pefile  # Built into CAPE to parse Windows executables
from lib.common.abstracts import Package


class BinSim(Package):
    """BinSim Execution Package: WoW64 Auto-Arch + Sysnative DLL Support + 64-bit Server Bridge"""

    TRACE_PATH = r"C:\Users\Public\binsim_trace.json"

    def check(self):
        """Heartbeat check: Force a graceful exit after 60 seconds of tracing!"""
        if hasattr(self, "start_time"):
            if time.time() - self.start_time > 60:
                print(
                    "[*] 60 seconds reached! Initiating graceful shutdown to save traces...",
                    flush=True,
                )
                return False  # Returning False tells CAPE to stop and run finish()!
        return True

    def start(self, path):
        self.start_time = time.time()  # Start the stopwatch!
        print("[*] BinSim Package Started...", flush=True)

        package_dir = os.path.dirname(os.path.abspath(__file__))
        bundled_js = os.path.normpath(
            os.path.join(package_dir, "..", "..", "data", "binsim_stalker.js")
        )
        guest_js_path = r"C:\binsim_stalker.js"

        try:
            if not os.path.exists(bundled_js):
                return None
            shutil.copy(bundled_js, guest_js_path)

            if os.path.exists(self.TRACE_PATH):
                try:
                    os.remove(self.TRACE_PATH)
                except Exception:
                    pass
        except Exception as e:
            return None

        try:
            import frida

            with open(guest_js_path, "r", encoding="utf-8") as f:
                js_code = f.read()

            # --- SMART ARCHITECTURE & DLL DETECTION ---
            try:
                pe = pefile.PE(path)
                # 0x8664 is the magic number for AMD64 (64-bit)
                is_64bit = pe.FILE_HEADER.Machine == 0x8664
                is_dll = pe.is_dll()
                print(
                    f"[*] Target PE Analysis -> 64-bit: {is_64bit} | Is DLL: {is_dll}",
                    flush=True,
                )
            except Exception as e:
                print(f"[-] PE Parse Error: {e}", flush=True)
                is_64bit, is_dll = False, False

            spawn_target = [path]

            # If it's a DLL, we MUST use rundll32.exe to spawn it!
            if is_dll:
                if is_64bit:
                    # THE FIX: 32-bit Python on 64-bit OS redirects System32 to SysWOW64.
                    # We MUST use the hidden 'Sysnative' alias to reach the real 64-bit rundll32.exe!
                    if os.path.exists(r"C:\Windows\Sysnative\rundll32.exe"):
                        rundll32 = r"C:\Windows\Sysnative\rundll32.exe"
                    else:
                        rundll32 = r"C:\Windows\System32\rundll32.exe"
                else:
                    # 32-bit rundll32 on a 64-bit OS lives in SysWOW64
                    if os.path.exists(r"C:\Windows\SysWOW64\rundll32.exe"):
                        rundll32 = r"C:\Windows\SysWOW64\rundll32.exe"
                    else:
                        rundll32 = r"C:\Windows\System32\rundll32.exe"

                # Attempt to run the default entry point (#1) of the DLL
                spawn_target = [rundll32, f"{path},#1"]
                print(
                    f"[*] DLL Detected! Spawning via host: {spawn_target}", flush=True
                )

            # --- THE 64-BIT SERVER BRIDGE ---
            print(
                f"[*] Attempting to spawn process via 64-bit Frida Server: {spawn_target}",
                flush=True,
            )

            try:
                # 1. Connect to the 64-bit frida-server listening in the background
                device = frida.get_device_manager().add_remote_device("127.0.0.1:27042")

                # 2. Tell the SERVER to spawn the malware!
                pid = device.spawn(spawn_target)
            except Exception as spawn_err:
                print(f"[-] Frida Server Spawn Failed: {spawn_err}", flush=True)
                err_str = str(spawn_err)

                # Catch 16-bit DOS / Corrupted Headers
                if "0x00000032" in err_str or "0x32" in err_str:
                    print(
                        "[!] OS REJECTION 0x32: This executable is fundamentally incompatible with the current Windows environment (e.g., 16-bit DOS or corrupted payload).",
                        flush=True,
                    )

                print("[*] Falling back to standard CAPE execution.", flush=True)
                return None  # Gracefully let CAPE's native monitor take over

            # 3. Use the DEVICE object to attach and create the script (NOT frida.attach)
            self.frida_session = device.attach(pid)
            self.frida_script = self.frida_session.create_script(js_code)

            # --- THE INFINITE LOOP FIX: PYTHON WRITES THE FILE ---
            def on_message(message, data):
                if message["type"] == "send":
                    payload = message.get("payload")
                    # If JavaScript sent a JSON dictionary, save it to the file
                    if (
                        isinstance(payload, dict)
                        and payload.get("type") == "binsim_trace"
                    ):
                        try:
                            with open(self.TRACE_PATH, "a", encoding="utf-8") as f:
                                f.write(json.dumps(payload["data"]) + "\n")
                                f.flush()  # Force Python buffer flush
                                os.fsync(f.fileno())  # FORCE OS TO WRITE TO DISK
                        except Exception as e:
                            print(f"[-] Python I/O Error: {e}", flush=True)
                    else:
                        # Otherwise, it's just a normal debug log
                        print(f"[FRIDA-MSG] {payload}", flush=True)
                elif message["type"] == "error":
                    # Catch anti-debugging crashes gracefully instead of breaking the script
                    print(
                        f"[-] Malware attacked the Frida Agent: {message}", flush=True
                    )

            self.frida_script.on("message", on_message)
            self.frida_script.load()

            print("[*] Resuming process execution via Server...", flush=True)
            # 4. Use the DEVICE object to resume (NOT frida.resume)
            device.resume(pid)

            # --- THE SHIELD FIX ---
            # Give the malware 5 seconds of pure, un-collided Frida execution!
            time.sleep(5)

            # --- THE LIFECYCLE FIX ---
            # Returning PID tells CAPE the process is alive.
            print(f"[*] Returning PID {pid} to CAPE. VM will stay alive.", flush=True)
            return pid

        except Exception as e:
            print(f"[-] CRITICAL NATIVE FRIDA CRASH: {e}", flush=True)
            return None

    def finish(self):
        print("[*] BinSim finish() called. Initiating trace upload...", flush=True)
        if not os.path.exists(self.TRACE_PATH):
            return

        try:
            from lib.common.results import upload_to_host

            upload_to_host(self.TRACE_PATH, "files/binsim_trace.json")
            print("[*] Uploaded BinSim trace via upload_to_host.", flush=True)
            return
        except Exception as e:
            pass
