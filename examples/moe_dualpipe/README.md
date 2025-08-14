### Command

```bash
MASTER_ADDR=localhost MASTER_PORT=12355 WORLD_SIZE=4 python examples/moe_train_basic.py
```

### Parameters
- WORLD_SIZE=4: Uses 4 GPUs for pipeline parallelism
- MASTER_ADDR: Master node address
- MASTER_PORT: Communication port

## Code Examples

The example includes two main test functions:

1. `test_moe_basic()`: Tests basic functionality of the MoE model

2. Example command to run:
```bash
MASTER_ADDR=localhost MASTER_PORT=12355 WORLD_SIZE=4 python examples/moe_train_basic.py
```



