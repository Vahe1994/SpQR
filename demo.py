import torch
import sys




if __name__ == '__main__':
    k = 2
    device = 'cuda'
    for m in [4096]:
        for n in [4096]:
            for flag in [False, True]:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flag
                torch.backends.cudnn.allow_tf32 = flag
                torch.backends.cuda.matmul.allow_tf32 = flag
                torch.backends.cudnn.allow_tf32 = flag


                d = torch.zeros((m, n), dtype=torch.float16, device=device)
                x = torch.zeros((n, k), dtype=torch.float16, device=device)
                y = torch.zeros((m, k), dtype=x.dtype, device=x.device)
                torch.matmul(d, x, out=y)