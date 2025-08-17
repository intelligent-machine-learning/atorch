
 

## Clone the DualPipe & Setup Environment

```bash
git clone https://github.com/deepseek-ai/DualPipe.git
cd dualpipe
conda create -n dualpipe python=3.10 -y
conda activate dualpipe
pip install -r requirements.txt
pip install -e .
```

## Naive Implementation for Single-GPU and Multi-GPU Training of MoE Models
```bash
MASTER_ADDR=localhost MASTER_PORT=12355 WORLD_SIZE=4 python examples/moe_train_basic.py
```

### Parameters
- WORLD_SIZE=4: Uses 4 GPUs for pipeline parallelism
- MASTER_ADDR: Master node address
- MASTER_PORT: Communication port
- `test_moe_basic()`: Tests basic functionality of the MoE model




