# Baseline Adversarial Techniques

This directory contains implementations of various **baseline adversarial techniques** for malware evasion. In this research repository, these techniques act strictly as **comparators** to evaluate the effectiveness of our proposed method (VERITAS). 

Each subdirectory within this folder corresponds to a specific baseline evasion method and is fully self-contained with its own detailed documentation.

## Included Baseline Methods

- **AIMED-RL**: An adversarial technique utilizing Reinforcement Learning to intelligently construct evasive malware mutations. Please see `AIMED-RL/README.md` for execution instructions.
- **DQEAF**: Deep Q-Learning for Evasion Attacks Framework. A reinforcement learning approach that learns sequences of functionality-preserving transformations to bypass detection. Detailed in `DQEAF/README.md`.
- **GAPGAN**: Generative Adversarial Payload GAN. Uses Generative Adversarial Networks to generate evasive payloads that mimic benign software distributions. See `GAPGAN/README.md`.
- **MAB-malware**: Multi-Armed Bandit approach for malware evasion. Treats the selection of obfuscation actions as a bandit problem to maximize evasion probability. Check `MAB-malware/README.md`.
- **MalGuise**: A technique focused on synthesizing API sequences to disguise malicious behavior dynamically. Instructions available in `MalGuise/README.md`.
- **OBFU-mal**: A comprehensive framework for malware obfuscation leveraging various structural and behavioral transformations. Refer to `OBFU-mal/README.md` for usage.
- **gamma**: Implementation of the GAMMA (Genetic Adversarial Malware Modification Algorithm) technique. 
  - *Note:* GAMMA requires the ML server (located in the `baseline_detectors` directory) to be running as its surrogate model. Please see the [GAMMA README](./gamma/README.md) for detailed instructions on starting the server.
- **malgpt**: Implementation of the MalGPT adversarial technique, leveraging language models to generate stealthy adversarial code sequences. See [MalGPT README](./malgpt/README.md).

## Requirements
While each technique may have specific dependencies listed in its respective directory, you can find the core dependencies required across these methods in `requirements.txt`.
