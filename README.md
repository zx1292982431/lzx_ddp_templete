# 16-mixed ddp train templete

How to start:
```
torchrun --nproc_per_node={num of gpus on current node} --nnodes={num total nodes} --node_rank={0 for master,i for others} --master_addr="{master host or ip adress}" --master_port={a free port of master} train_ddp.py --train_script_arg1 value_of_arg1 --train_script_arg2 value_of_arg2 ...
```
