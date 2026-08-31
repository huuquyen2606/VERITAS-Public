#!/usr/bin/env python3
"""
Welcome to the Framework for Adversarial Malware Evaluation (FAME)

FAME was designed to understand how byte-level transformations could automatically be injected to Windows Portable
Executable (PE) files and compromise ML-based malware classifiers. Moreover, it supports integrity verification to
ensure that the new adversarial examples are valid. This work implements the action space proposed on the OpenAI gym
malware environment. It has been implemented in Fedora 30 and tested on Ubuntu 16 using Python3. Library versions are
defined in requirements.txt file.

The following modules are available: ARMED, AIMED, AIMED-RL & GAME-UP

GAME-UP: Generating Adversarial Malware Examples with Universal Perturbations

This work intends to understand how Universal Adversarial Perturbations (UAPs) can be useful to create efficient
adversarial examples compared to input-specific attacks. Furthermore, it explores how real malware examples in the
problem-space affect the feature-space of classifiers to identify systematic weaknesses. Also, it implements a variant
of adversarial training to improve the resilience of static ML-based malware classifiers for Windows PE binaries.

AIMED-RL: Automatic Intelligent Modifications to Evade Detection (with Reinforcement Learning)

This work is focused on understanding how sensitive static malware classifiers are to adversarial examples. It uses
different techniques including Genetic Programming (GP) and Reinforcement Learning (RL) to inject perturbations to
Windows portable executable malware without compromising its functionality and, thus, keeping the new generated
adversarial example valid.

"""

import argparse
import time
import src.config as cfg
import src.functions as f
import src.implementation as i


def parse_args(argv=None):
	parser = argparse.ArgumentParser(description='Framework for Adversarial Malware Evaluation')
	parser.add_argument('module', help='Module to run, e.g. AIMED-RL')
	parser.add_argument('--train', dest='train', action='store_const', const=True, default=None,
					help='Train a new AIMED-RL agent')
	parser.add_argument('--no-train', dest='train', action='store_const', const=False,
					help='Skip AIMED-RL training')
	parser.add_argument('--evaluate', dest='evaluate', action='store_const', const=True, default=None,
					help='Evaluate an AIMED-RL agent')
	parser.add_argument('--no-evaluate', dest='evaluate', action='store_const', const=False,
					help='Skip AIMED-RL evaluation')
	parser.add_argument('--train-data', dest='train_data_path',
					help='Training dataset root for AIMED-RL. Recursive traversal is supported.')
	parser.add_argument('--eval-data', dest='evaluation_data_path',
					help='Evaluation dataset root for AIMED-RL. Recursive traversal is supported.')
	parser.add_argument('--detector-path', dest='detector_path',
					help='Path to the LightGBM detector (.pkl or .txt)')
	parser.add_argument('--threshold', dest='threshold', type=float,
					help='Detection threshold used by AIMED-RL')
	parser.add_argument('--episodes', dest='episodes', type=int,
					help='Number of training episodes for AIMED-RL')
	parser.add_argument('--max-turns', dest='max_turns', type=int,
					help='Maximum number of turns for AIMED-RL')
	parser.add_argument('--recursive-data', dest='recursive_data', action='store_const', const=True, default=None,
					help='Read AIMED-RL datasets recursively')
	parser.add_argument('--no-recursive-data', dest='recursive_data', action='store_const', const=False,
					help='Read AIMED-RL datasets non-recursively')
	parser.add_argument('--agent-dir', dest='agent_dir',
					help='Directory of the AIMED-RL agent to evaluate')
	parser.add_argument('--agent-info', dest='agent_information',
					help='Training report CSV associated with the AIMED-RL agent')
	parser.add_argument('--save-ae-dir', dest='save_ae_dir',
					help='Directory used to save successful AIMED-RL adversarial examples')
	parser.add_argument('--save-train-aes', dest='save_train_aes', action='store_const', const=True, default=None,
					help='Save successful adversarial examples found during AIMED-RL training')
	parser.add_argument('--no-save-train-aes', dest='save_train_aes', action='store_const', const=False,
					help='Do not save successful adversarial examples found during AIMED-RL training')
	parser.add_argument('--save-eval-aes', dest='save_eval_aes', action='store_const', const=True, default=None,
					help='Save successful adversarial examples found during AIMED-RL evaluation')
	parser.add_argument('--no-save-eval-aes', dest='save_eval_aes', action='store_const', const=False,
					help='Do not save successful adversarial examples found during AIMED-RL evaluation')
	return parser.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	option = args.module.upper().replace('_', '-')

	# Time algorithm
	start = time.time()

	# ARMED: Finding adversarial malware examples stochastically
	if option == 'ARMED':
		i.armed(number_perturbations=cfg.file.getint('armed', 'perturbations'),
				rounds=cfg.file.getint('armed', 'rounds'), files_expected=cfg.file.getint('armed', 'advFilesExpected'),
				model=cfg.file['armed']['model'])

	# ARMED II: Using Incremental Iterations of perturbations' sequence
	elif option == 'ARMED-II':
		i.armed2(number_perturbations=cfg.file.getint('armed', 'perturbations'),
				 rounds=cfg.file.getint('armed', 'rounds'),
				 files_expected=cfg.file.getint('armed', 'advFilesExpected'),
				 model=cfg.file['armed']['model'])

	# AIMED: Finding adversarial examples with genetic programming
	elif option == 'AIMED':
		i.aimed(size_population=cfg.file.getint('aimed', 'sizePopulation'),
				number_perturbations=cfg.file.getint('aimed', 'perturbations'),
				model=cfg.file['aimed']['model'])

	# AIMED-RL: Finding adversarial examples with reinforcement learning
	elif option in ('AIMED-RL', 'AIMEDRL'):
		i.aimed_rl(base_path=cfg.file['paths']['rl'],
				   report_path=cfg.file['paths']['report'],
				   train=args.train,
				   evaluate=args.evaluate,
				   train_data_path=args.train_data_path,
				   evaluation_data_path=args.evaluation_data_path,
				   detector_path=args.detector_path,
				   threshold=args.threshold,
				   episodes=args.episodes,
				   max_turns=args.max_turns,
				   recursive_data=args.recursive_data,
				   agent_dir=args.agent_dir,
				   agent_information=args.agent_information,
				   save_ae_dir=args.save_ae_dir,
				   save_train_aes=args.save_train_aes,
				   save_eval_aes=args.save_eval_aes)

	# GAME-UP: Find universal perturbation sequences to generate adversarial examples
	elif option in ('GAMEUP', 'GAME-UP'):
		i.gameup(number_perturbations=cfg.file.getint('gameup', 'perturbations'), model=cfg.file['gameup']['model'],
				 exploration_set=cfg.file['paths']['exploration'],)

	# UAP-DEF: Use UAPs to increase resilience of models against universal attacks
	elif option == 'DEFENSE':
		i.defense(number_perturbations=cfg.file.getint('defense', 'perturbations'),
				  model=cfg.file['defense']['model'])

	# COMPARE: Evaluate different algorithms (Example imp.: AIMED vs ARMED)
	elif option == 'COMPARE':
		i.comparing(number_perturbations=cfg.file.getint('compare', 'perturbations'),
					rounds=cfg.file.getint('compare', 'rounds'),
					files_expected=cfg.file.getint('compare', 'advFilesExpected'),
					model=cfg.file['compare']['model'])

	else:
		exit('Option not found!')

	f.time_me(start)


if __name__ == '__main__':
	main()
