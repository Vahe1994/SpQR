import inspect
from typing import List
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 


def collate_fn(batch):
    return pad_sequence([x[0] for x in batch], batch_first=True).squeeze(1)


def collate_fn_with_teacher_outputs(batch):
    return (
        pad_sequence([x[0] for x in batch], batch_first=True).squeeze(1),
        torch.cat([x[1] for x in batch], dim=0)
    )


class InputCollector(nn.Module):
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.inputs = []
        self.input_kwargs = []

    def forward(self, inputs=None, **input_kwargs):
        """
        Assumes that the wrapped module has a single 
        input that can reside in inputs or input_kwargs.
        """
        # find required forward args among
        if inputs is None:
            input_found = False
            forward_params = inspect.signature(self.module.forward).parameters
            for k, v in forward_params.items():
                if v.default is inspect._empty:
                    inputs = input_kwargs[k]
                    input_found = True
                    del input_kwargs[k]
                    break
            if not input_found:
                raise RuntimeError("Input not found")

        self.inputs.append(inputs)
        self.input_kwargs.append(input_kwargs)
        raise ValueError


@torch.no_grad()
def cache_teacher_outputs(
    model: nn.Module, 
    data, 
    target_feature: str,
    blocks_name: str,
    pre_blocks_modules: List[str] = [],
    post_blocks_modules: List[str] = [],
    device: str = 'cuda'
):
    device = torch.device(device)
    blocks = model.get_submodule(blocks_name)
    blocks[0] = blocks[0].to(device)
    # load input embeddings or any other preprocessing step
    for module_name in pre_blocks_modules:
        module = model.get_submodule(module_name)
        module.to(device)

    ### Input preparation ### 
    blocks[0] = InputCollector(blocks[0])
    for batch in data:
        try:
            model(batch.to(device))
        except ValueError:
            pass
    inputs = blocks[0].inputs
    input_kwargs = blocks[0].input_kwargs
    blocks[0] = blocks[0].module

    for module_name in pre_blocks_modules:
        module = model.get_submodule(module_name)
        module.cpu()

    for i in range(len(blocks)):
        block = blocks[i].to(device)
        for inp_id, inp in enumerate(inputs):
            out = block(inp, **input_kwargs[inp_id])
            if isinstance(out, (list, tuple)):
                out = out[0]
            inp.data = out
        block = block.cpu()
    
    # output modules
    for module_name in post_blocks_modules:
        module = model.get_submodule(module_name)
        module = module.to(device)
        for inp_id, inp in enumerate(inputs):
            inp.data = module(inp)
        module = module.cpu()
        if module_name  == target_feature:
            break

    return [x.cpu() for x in inputs]
            

def train_student(
    model, 
    data, 
    teacher_outputs, 
    args
):
    device = next(iter(model.parameters())).device

    train_data_with_teacher_outputs = [
        (data[i][0], teacher_outputs[i]) for i in range(len(data))
    ]
    # prepate train loader
    train_loader = DataLoader(
        train_data_with_teacher_outputs,
        batch_size=args.finetune_batch_size, 
        num_workers=args.num_workers, 
        pin_memory=True, 
        collate_fn=collate_fn_with_teacher_outputs
    )

    cached_outputs = {}
    hook = None
    if args.target_feature:
        feature_module = model.get_submodule(args.target_feature)
        def cache_outputs_hook(module, inputs, outputs):
            cached_outputs['data'] = outputs
        hook = feature_module.register_forward_hook(cache_outputs_hook)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], 
        lr=args.lr
    )
    scaler = GradScaler()
    pbar = trange(args.finetune_num_steps, desc='Training')
    step = 0
    stop = False
    while not stop:
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            # forward pass and update cached_outputs
            with autocast(enabled=True, dtype=torch.float16):
                outputs = model(inputs).logits
                if args.target_feature:
                    loss = F.mse_loss(cached_outputs['data'], targets)    
                else:
                    loss = F.mse_loss(outputs, targets)
            if torch.isnan(loss):
                raise RuntimeError("NaN loss encountered")

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.update(1)
            step += 1
            if step % args.logging_steps == 0:
                print(f'Step {step} Loss {loss.item():.3f}')
            if step == args.finetune_num_steps:
                stop = True
                break

    if hook:
        hook.remove()