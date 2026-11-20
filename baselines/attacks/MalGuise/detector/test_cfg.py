import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'MAGIC' / 'maldefender'))
from cfg_builder import ControlFlowGraphBuilder
import glog as log
log.setLevel("DEBUG")

# filePrefix without .asm
builder = ControlFlowGraphBuilder('test_angr', str(BASE_DIR))
cfg = builder.getControlFlowGraph()
print(f"Nodes: {cfg.number_of_nodes()}, Edges: {cfg.number_of_edges()}")
