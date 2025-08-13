import time
from typing import Callable

import torch
from torch._C._profiler import ProfilerActivity
from cs336_basics_lx.cs336_basics.transformer import Linear,MultiHeadSelfAttention, TransformerBlock, Transformer, softmax

from triton_impl import triton_gelu,print_ptx, triton_softmax

def benchmark(description: str, run: Callable, num_warmups: int=1, num_trials: int=1):


    for _ in range(num_warmups):

        run()


    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total_time = 0
    for i in range(num_trials):
        start_time = time.time()

        run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time += (time.time() - start_time) * 1000

    return total_time / num_trials


def profile(description: str, run: Callable, num_warmups: int = 1, with_stack: bool = True):

    for _ in range(num_warmups):
        run()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=with_stack,
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)
    ) as prof:
        run()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=80, row_limit=10)

    if with_stack:
        text_path = f"var/stacks_{description}.txt"
        svg_path = f"var/stacks_{description}.svg"
        prof.export_stacks(text_path, "self_cuda_time_total")

    return table


def run_operation(dim: int, operation: Callable) -> Callable:

    x = torch.randn(dim, dim).to("cuda:0")
    return lambda : operation(x)


def run_operation2(dim: int, operation: Callable) -> Callable:

    x = torch.randn(dim, dim, device="cuda:0")
    y = torch.randn(dim, dim, device="cuda:0")
    return lambda : operation(x, y)


def triton_gelu_main():

    if not torch.cuda.is_available():
        return

    # y1 = triton_gelu(x)
    #
    # print_ptx("triton_gelu", triton_gelu_kernel)

    print(f"triton_kernel: \n{profile("triton_gelu", run=run_operation(dim=1024, operation=triton_gelu))}")

if __name__ == '__main__':
    #
    # avg_time = benchmark("sleep", lambda : time.sleep( 50 / 1000 ), num_warmups=1, num_trials=10)
    # print(f"avg_time: {avg_time}")

    sleep_function = lambda : time.sleep(50 / 1000)
    sleep_profile = profile("sleep", sleep_function)
    print(f"Sleep:\n {sleep_profile}")

    add_function = lambda a, b: a + b
    print(f"Add:\n {profile("add", run_operation2(dim=2048, operation=add_function))}")


    matmul_function = lambda a, b: a @ b
    print(f"matmul:\n {profile("matmul", run_operation2(dim=2048, operation=matmul_function))}")

    cdist_function = lambda a, b: torch.cdist(a, b)
    print(f"cdist:\n {profile("cdist", run_operation2(dim=2048, operation=cdist_function))}")


    gelu_function = lambda a, b: torch.nn.functional.gelu(a +  b)
    print(f"gelu:\n {profile("gelu", run_operation2(dim=2048, operation=gelu_function))}")


    softmax_function = lambda a, b: torch.nn.functional.softmax(a +  b, dim=-1)
    print(f"softmax:\n {profile("softmax", run_operation2(dim=2048, operation=softmax_function))}")

    # transformer = Transformer(d_model=512, num_heads=8, vocab_size=512, context_length=512, d_ff=16, theta=5,
    #                           use_rope=True, num_layers=4, device="cuda:0")
    #
    # input_ = torch.randint(low=0, high=512, size=(32, 64)).to("cuda:0")
    # run_transformer = lambda a, b: transformer(input_)
    #
    # print(f"transformer:\n {profile("transformer", run_operation2(dim=2048, operation=run_transformer))}")
    # kernel
    # triton_gelu_main()

    print(f"triton_gelu:\n {profile("triton_gelu", run=run_operation(dim=1024, operation=triton_gelu))}")

    pytorch_gelu = lambda a : torch.nn.functional.gelu(a)
    print(f"pytorch gelu:\n {profile("pytorch gelu", run_operation(dim=1024, operation=pytorch_gelu))}")


    #
    # x = torch.tensor([
    #     [5., 5., 5.],
    #     [0, 0, 100],
    # ], device="cuda:0")
    #
    # y1 = softmax(x)
    #
    # y2 = triton_softmax(x)
    # print(y1, y2)
    # assert torch.allclose(y1, y2)

    print(f"softmax_kernel:\n {profile("triton_softmax", run=run_operation(dim=2048, operation=triton_softmax))}")

    print(f"pytorch softmax:\n {profile("pytorch_softmax", run=run_operation(dim=2048, operation=softmax))}")