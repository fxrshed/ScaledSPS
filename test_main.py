def test_sample():
    assert 1 == 2

# w = torch.zeros(train_data.shape[1], device=device).requires_grad_()
# z = rademacher(train_data.shape[1]).to(device=device).requires_grad_()


# loss = F(w)
# grad, = torch.autograd.grad(loss, w, create_graph=True)
# H = torch.autograd.functional.hessian(F, w)
# Hvp = torch.autograd.functional.hvp(F, w, z)[1]
# Hvp

# g = torch.autograd.grad(loss, w, create_graph=True)[0]

# Ds = []
# for j in range(10000):
#     z = rademacher(train_data.shape[1]).to(device=device).requires_grad_()
#     hvp = torch.autograd.grad(g,  w,  grad_outputs = z, retain_graph = True)[0]
#     Ds.append( (hvp*z) )

# H_diag = torch.mean(torch.stack(Ds), 0) 
# assert torch.abs(H.diagonal().sum() - H_diag.sum()) < 0.1