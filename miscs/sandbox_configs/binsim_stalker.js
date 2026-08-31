// binsim_stalker.js
// Phase 24: The Mathematically Pure Engine (Strict Syscall Mapping & Win10 Dropper Catching)

var MAX_INSTRUCTIONS = 250;
var threadTrace = {};
var stalkedThreads = {};
var isTracingActive = false;

// --- TRITON BACKEND TRACKING ---
var globalSyscallIndex = 0;
var previousReturns = [];

// --- THE STRICT ACADEMIC SYSCALL MAPPING ---
// These perfectly match the CRITICAL_SYSCALLS set in binsim_align.py!
var apiToId = {
  // File Operations
  NtCreateFile: 0x42,
  NtOpenFile: 0x30,
  NtClose: 0x0c,
  NtWriteFile: 0x112,
  NtReadFile: 0xfe,
  NtDeleteFile: 0x1d,
  NtQueryDirectoryFile: 0x0f,
  NtSetInformationFile: 0x3f,
  NtQueryInformationFile: 0x5d,

  // Memory & Injection
  NtAllocateVirtualMemory: 0x15,
  NtWriteVirtualMemory: 0x37,
  NtProtectVirtualMemory: 0x4d,
  NtUnmapViewOfSection: 0x10a,

  // Process & Thread Hijacking
  NtCreateProcess: 0x2f,
  NtCreateProcessEx: 0x2f,
  NtCreateUserProcess: 0x2f, // Map modern Win10 API to the same behavior
  NtOpenProcess: 0x23,
  NtTerminateProcess: 0xce,
  NtCreateThread: 0x114,
  NtCreateThreadEx: 0x114,

  // Evasion & Sleep
  NtDelayExecution: 0x35,
  NtCreateMutant: 0x13,

  // Registry Operations
  NtCreateKey: 0x12a,
  NtOpenKey: 0x07,
  NtSetValueKey: 0xd5,
};

send("[+] BinSim Frida Payload Injected. Waiting for OS Loader to finish...");

function writeTrace(obj) {
  send({ type: "binsim_trace", data: obj });
}

writeTrace({ event: "payload_loaded", ts: Date.now() });

function getThreadBuffer(tid) {
  if (!threadTrace[tid]) {
    threadTrace[tid] = {
      buffer: new Array(MAX_INSTRUCTIONS),
      index: 0,
      count: 0,
    };
  }
  return threadTrace[tid];
}

function getTrace(tid) {
  var tb = getThreadBuffer(tid);
  var result = [];
  for (var i = 0; i < tb.count; i++) {
    var pos = (tb.index - tb.count + i + MAX_INSTRUCTIONS) % MAX_INSTRUCTIONS;
    result.push(tb.buffer[pos]);
  }
  return result;
}

// --- MODULE FILTER (Whitelist Approach) ---
var appModules = [];
function updateModules() {
  appModules = [];
  Process.enumerateModules().forEach(function (m) {
    if (m.path && m.path.toLowerCase().indexOf("c:\\windows\\") === -1) {
      appModules.push({ base: m.base, end: m.base.add(m.size) });
    }
  });
}
updateModules();

function isAppCode(addr) {
  for (var i = 0; i < appModules.length; i++) {
    if (
      addr.compare(appModules[i].base) >= 0 &&
      addr.compare(appModules[i].end) < 0
    ) {
      return true;
    }
  }
  return false;
}

// --- GUI-SAFE JIT STALKER ---
function attachStalker(tid) {
  if (stalkedThreads[tid]) return;
  stalkedThreads[tid] = true;

  send("[+] Attaching GUI-Safe JIT Stalker to thread: " + tid);
  try {
    Stalker.follow(tid, {
      events: {
        call: false,
        ret: false,
        exec: false,
        block: false,
        compile: true,
      },
      transform: function (iterator) {
        var instruction = iterator.next();
        if (instruction === null) return;

        var isMalwareBlock = isAppCode(instruction.address);

        while (instruction !== null) {
          if (isMalwareBlock) {
            var currentTid = Process.getCurrentThreadId();
            var tb = getThreadBuffer(currentTid);

            // DEFERRED BYTE READING: Save pointers now to save CPU!
            tb.buffer[tb.index] = {
              _pc_ptr: instruction.address,
              _size: instruction.size,
              pc: safeHexPointer(instruction.address),
              mnemonic: instruction.mnemonic,
              opStr: instruction.opStr,
              bytes: "",
              regs: {},
            };
            tb.index = (tb.index + 1) % MAX_INSTRUCTIONS;
            if (tb.count < MAX_INSTRUCTIONS) tb.count++;
          }
          iterator.keep();
          instruction = iterator.next();
        }
      },
    });
  } catch (e) {}
}

var TARGETS = [
  { mod: "ntdll.dll", api: "NtCreateFile" },
  { mod: "ntdll.dll", api: "NtOpenFile" },
  { mod: "ntdll.dll", api: "NtWriteFile" },
  { mod: "ntdll.dll", api: "NtReadFile" },
  { mod: "ntdll.dll", api: "NtClose" },
  { mod: "ntdll.dll", api: "NtDeleteFile" },
  { mod: "ntdll.dll", api: "NtQueryDirectoryFile" },
  { mod: "ntdll.dll", api: "NtSetInformationFile" },
  { mod: "ntdll.dll", api: "NtQueryInformationFile" },
  { mod: "ntdll.dll", api: "NtQueryAttributesFile" },
  { mod: "ntdll.dll", api: "NtQueryFullAttributesFile" },
  { mod: "kernel32.dll", api: "CopyFileA" },
  { mod: "kernel32.dll", api: "CopyFileW" },
  { mod: "kernel32.dll", api: "MoveFileA" },
  { mod: "kernel32.dll", api: "MoveFileW" },
  { mod: "ntdll.dll", api: "NtCreateKey" },
  { mod: "ntdll.dll", api: "NtOpenKey" },
  { mod: "ntdll.dll", api: "NtSetValueKey" },
  { mod: "ntdll.dll", api: "NtQueryValueKey" },
  { mod: "ntdll.dll", api: "NtSaveKey" },
  { mod: "ntdll.dll", api: "NtAllocateVirtualMemory" },
  { mod: "ntdll.dll", api: "NtProtectVirtualMemory" },
  { mod: "ntdll.dll", api: "NtMapViewOfSection" },
  { mod: "ntdll.dll", api: "NtWriteVirtualMemory" },
  { mod: "ntdll.dll", api: "NtReadVirtualMemory" },

  // --- ADDED MISSING WIN10 PROCESS AND MEMORY APIs ---
  { mod: "ntdll.dll", api: "NtCreateUserProcess" },
  { mod: "ntdll.dll", api: "NtUnmapViewOfSection" },

  { mod: "ntdll.dll", api: "NtCreateProcess" },
  { mod: "ntdll.dll", api: "NtCreateProcessEx" },
  { mod: "ntdll.dll", api: "NtOpenProcess" },
  { mod: "ntdll.dll", api: "NtTerminateProcess" },
  { mod: "ntdll.dll", api: "NtCreateThread" },
  { mod: "ntdll.dll", api: "NtCreateThreadEx" },
  { mod: "ntdll.dll", api: "NtResumeThread" },
  { mod: "ntdll.dll", api: "NtTerminateThread" },
  { mod: "ntdll.dll", api: "NtGetContextThread" },
  { mod: "ntdll.dll", api: "NtSetContextThread" },
  { mod: "ntdll.dll", api: "NtQueueApcThread" },
  { mod: "ntdll.dll", api: "NtDelayExecution" },
  { mod: "ntdll.dll", api: "NtQuerySystemInformation" },
  { mod: "ntdll.dll", api: "NtQueryInformationProcess" },
  { mod: "ntdll.dll", api: "NtCreateMutant" },
  { mod: "ntdll.dll", api: "NtOpenMutant" },
  { mod: "ws2_32.dll", api: "connect" },
  { mod: "ws2_32.dll", api: "bind" },
  { mod: "ws2_32.dll", api: "send" },
  { mod: "ws2_32.dll", api: "recv" },
  { mod: "ws2_32.dll", api: "gethostname" },
  { mod: "wininet.dll", api: "InternetOpenUrlA" },
  { mod: "wininet.dll", api: "InternetOpenUrlW" },
  { mod: "wininet.dll", api: "HttpSendRequestA" },
  { mod: "wininet.dll", api: "HttpSendRequestW" },
  { mod: "kernel32.dll", api: "LoadLibraryA" },
  { mod: "kernel32.dll", api: "LoadLibraryW" },
  { mod: "kernel32.dll", api: "GetProcAddress" },
  { mod: "kernel32.dll", api: "GetModuleHandleA" },
];

var pendingTargets = TARGETS.slice();
var hooksPlaced = 0;
var hooked = {};

function findExport(moduleName, exportName) {
  try {
    var m = Process.findModuleByName(moduleName);
    if (m && typeof m.findExportByName === "function")
      return m.findExportByName(exportName);
  } catch (e) {}
  try {
    var m2 = Process.getModuleByName(moduleName);
    if (m2 && typeof m2.getExportByName === "function")
      return m2.getExportByName(exportName);
  } catch (e) {}
  return null;
}

function safeHexPointer(p) {
  try {
    return p.toString();
  } catch (e) {
    return "null";
  }
}

function doHook(targetAddress, target) {
  var key = target.mod + "!" + target.api;
  if (hooked[key]) return;

  try {
    Interceptor.attach(targetAddress, {
      onEnter: function (args) {
        if (!isTracingActive) return;

        if (target.api === "NtAllocateVirtualMemory") {
          this.allocBasePtr = args[1];
          this.allocSizePtr = args[3];
          this.protect = args[5].toInt32();
        }

        var tid = 0;
        try {
          tid = Process.getCurrentThreadId();
        } catch (e) {}

        var preBufferSlice = getTrace(tid);
        var currentPc = safeHexPointer(
          this.context.pc ||
            this.context.eip ||
            this.context.rip ||
            targetAddress,
        );

        var ctx = {};
        try {
          var regs = [
            "pc",
            "sp",
            "eax",
            "ebx",
            "ecx",
            "edx",
            "esi",
            "edi",
            "ebp",
            "eip",
            "rax",
            "rbx",
            "rcx",
            "rdx",
            "rsi",
            "rdi",
            "rbp",
            "rsp",
            "rip",
            "eflags",
          ];
          for (var i = 0; i < regs.length; i++) {
            var regName = regs[i];
            if (this.context[regName] !== undefined) {
              ctx[regName] = safeHexPointer(this.context[regName]);
            }
          }
        } catch (e) {
          ctx = { error: "context serialize failed: " + e };
        }

        this.traceEvent = {
          api: target.api,
          pc: currentPc,
          tid: tid,
          args: [
            safeHexPointer(args[0]),
            safeHexPointer(args[1]),
            safeHexPointer(args[2]),
            safeHexPointer(args[3]),
            safeHexPointer(args[4]),
            safeHexPointer(args[5]),
          ],
          context: ctx,
          pre_instructions: preBufferSlice,
        };
      },

      onLeave: function (retval) {
        if (!isTracingActive) return;

        if (
          target.api === "NtAllocateVirtualMemory" &&
          retval.toInt32() === 0
        ) {
          if (this.protect === 0x40 || this.protect === 0x20) {
            try {
              var allocatedBase = this.allocBasePtr.readPointer();
              var allocatedSize = this.allocSizePtr.readPointer();
              appModules.push({
                base: allocatedBase,
                end: allocatedBase.add(allocatedSize),
              });
            } catch (e) {}
          }
        }

        if (this.traceEvent) {
          // 1. Zero Buffer Check
          if (this.traceEvent.pre_instructions.length === 0) {
            return;
          }

          // 2. STRICT ACADEMIC FILTER: If this isn't a critical syscall required by Triton, DROP IT NOW!
          var apiId = apiToId[target.api];
          if (!apiId) {
            return;
          }

          // --- DEFERRED BYTE READING EXECUTES HERE ---
          for (var k = 0; k < this.traceEvent.pre_instructions.length; k++) {
            var inst = this.traceEvent.pre_instructions[k];
            if (!inst.bytes && inst._pc_ptr) {
              try {
                var buf = inst._pc_ptr.readByteArray(inst._size);
                if (buf) {
                  var u8 = new Uint8Array(buf);
                  var rawBytes = "";
                  for (var b = 0; b < u8.length; b++) {
                    var hex = u8[b].toString(16);
                    rawBytes += hex.length === 1 ? "0" + hex : hex;
                  }
                  inst.bytes = rawBytes;
                }
              } catch (e) {}
            }
            delete inst._pc_ptr;
            delete inst._size;
          }

          var retHex = safeHexPointer(retval);
          var currentIdx = globalSyscallIndex++;

          if (this.traceEvent.pre_instructions.length > 0) {
            this.traceEvent.pre_instructions[0].regs = this.traceEvent.context;
          }

          var deps = [];
          for (var d = 0; d < previousReturns.length; d++) {
            for (var a = 0; a < this.traceEvent.args.length; a++) {
              if (
                this.traceEvent.args[a] === previousReturns[d].val &&
                previousReturns[d].val !== "0x0"
              ) {
                deps.push({ from_syscall_idx: previousReturns[d].idx });
                break;
              }
            }
          }

          var finalPayload = {
            id: apiId,
            api: this.traceEvent.api,
            pc: this.traceEvent.pc,
            args: this.traceEvent.args,
            ret: retHex,
            deps: deps,
            instruction_segment: this.traceEvent.pre_instructions,
            memory_log: [],
          };

          writeTrace(finalPayload);

          if (retHex !== "0x0" && retHex !== "0x1") {
            previousReturns.push({ idx: currentIdx, val: retHex });
          }

          if (threadTrace[this.traceEvent.tid]) {
            threadTrace[this.traceEvent.tid].index = 0;
            threadTrace[this.traceEvent.tid].count = 0;
          }
        }

        if (
          target.api === "NtCreateThread" ||
          target.api === "NtCreateThreadEx"
        ) {
          setTimeout(function () {
            var currentThreads = Process.enumerateThreads();
            for (var t = 0; t < currentThreads.length; t++)
              attachStalker(currentThreads[t].id);
          }, 500);
        }
      },
    });
    hooked[key] = true;
    hooksPlaced += 1;
  } catch (e) {}
}

function tryHooking() {
  var remaining = [];
  for (var i = 0; i < pendingTargets.length; i++) {
    var target = pendingTargets[i];
    try {
      var addr = findExport(target.mod, target.api);
      if (addr) {
        doHook(addr, target);
      } else {
        remaining.push(target);
      }
    } catch (e) {
      remaining.push(target);
    }
  }
  pendingTargets = remaining;
}
// ==========================================
// --- PHASE 25: DYNAMIC & TIERED INJECTION ---
// ==========================================

// 1. DYNAMIC TRIGGER: Catch Late-Loading DLLs
try {
  var ldrAddr = Module.findExportByName("ntdll.dll", "LdrLoadDll");
  if (ldrAddr) {
    Interceptor.attach(ldrAddr, {
      onLeave: function (retval) {
        // Every time a new DLL loads, refresh our whitelists and try hooking missing APIs!
        updateModules();
        tryHooking();
      },
    });
    send("[+] LdrLoadDll dynamic trigger hooked.");
  }
} catch (e) {}

// 2. TIERED TIMEOUTS: The Safety Net
// Tier 1: 500ms - Catch fast malware, droppers, and unpacked stubs instantly
setTimeout(function () {
  send("[+] Tier 1 (500ms): Catching fast execution...");
  updateModules();
  tryHooking();

  var initialThreads = Process.enumerateThreads();
  for (var t = 0; t < initialThreads.length; t++)
    attachStalker(initialThreads[t].id);

  isTracingActive = true; // Turn on recording immediately!
}, 500);

// Tier 2: 1500ms - Catch standard applications and delayed packers
setTimeout(function () {
  send("[+] Tier 2 (1500ms): Refreshing hooks for standard apps...");
  updateModules();
  tryHooking();
}, 1500);

// Tier 3: 4000ms - Catch slow GUI apps (like PuTTY) after the window draws
setTimeout(function () {
  send("[+] Tier 3 (4000ms): Final pass for slow GUI initialization...");
  updateModules();
  tryHooking();
  send("[+] Total Hooks Placed across all tiers: " + hooksPlaced);
}, 4000);
