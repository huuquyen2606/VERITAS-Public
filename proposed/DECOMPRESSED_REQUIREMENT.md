# Decompression Requirements

To adhere to GitHub file size limitations (< 100 MB), large binary model weights are tracked as compressed `.xz` archives in the repository.

Before executing training or evaluation pipelines, please decompress the required asset(s):

## 1. Surrogate Detector Model Weights

Execute the following command from the `proposed/` directory:

```bash
unxz -k env/adv_RL_env/malware_rl_system/weights/m_attn_health_multi.pth.xz
```

This decompresses the pre-trained weights to `env/adv_RL_env/malware_rl_system/weights/m_attn_health_multi.pth` while keeping the original `.xz` archive intact.
