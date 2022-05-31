import torch
device = torch.device('cpu')
run_on_gpu = False
if torch.cuda.is_available():
    device = torch.device('cuda')
    run_on_gpu = True

x = torch.randn(2, 3, requires_grad=True)
y = torch.rand(2, 3, requires_grad=True)
z = torch.ones(2, 3, requires_grad=True)

with torch.autograd.profiler.profile(use_cuda=run_on_gpu) as prf:
    for _ in range(1000):
        z = (z / x) * y

print(prf.key_averages().table(sort_by='self_cpu_time_total'))


# Jacobian and Hessian matrices
def exp_adder(x, y):
    return 2 * x.exp() + 3 * y


inputs = (torch.rand(1), torch.rand(1))
print(torch.autograd.functional.jacobian(exp_adder, inputs))
