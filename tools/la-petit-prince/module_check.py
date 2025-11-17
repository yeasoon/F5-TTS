def check_conv_weight_norm():
    import torch
    import torch.nn as nn
    import torch.nn.utils as utils

    # Example Conv2d + weight_norm
    conv = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    wn_conv = utils.weight_norm(conv)

    # Random input
    x = torch.randn(1, 64, 32, 32)

    # Forward output from weight_norm layer
    

    # Get the "real" weight and bias as PyTorch computes
    with torch.no_grad():
        out_ref = wn_conv(x)
        # weight_real = wn_conv.weight_g * wn_conv.weight_v / wn_conv.weight_v.norm(dim=(1,2,3), keepdim=True)
        # bias_real = wn_conv.bias
        v_manual = wn_conv.weight_v
        g_manual = wn_conv.weight_g
        norm = v_manual.view(v_manual.size(0), -1).norm(dim=1, keepdim=True)  # shape: [out_c, 1]
        norm = norm.view(-1, 1, 1, 1)  # reshape to broadcast
        weight_manual = g_manual * v_manual/norm
        bias_manual = wn_conv.bias  # copy from wn_conv.bias

        # Do convolution manually
        out_manual = nn.functional.conv2d(x, weight_manual, bias_manual, stride=2, padding=1)
        # Mean absolute error
        diff = (out_ref - out_manual).abs().mean()
        print("Mean absolute difference:", diff.item())

        # Maximum difference
        print("Max absolute difference:", (out_ref - out_manual).abs().max().item())
def check_gru():
    import torch
    # class ManualGRU(torch.nn.Module):
    #     def __init__(self, input_size, hidden_size):
    #         super().__init__()
    #         self.input_size = input_size
    #         self.hidden_size = hidden_size

    #         # Match PyTorch's GRU gate order: [r, z, n]
    #         self.W_ih = torch.nn.Parameter(torch.randn(3 * hidden_size, input_size))
    #         self.W_hh = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
    #         self.b_ih = torch.nn.Parameter(torch.randn(3 * hidden_size))
    #         self.b_hh = torch.nn.Parameter(torch.randn(3 * hidden_size))

    #     def forward(self, x, h0=None):
    #         # x: [B, T, input_size]
    #         B, T, _ = x.shape
    #         if h0 is None:
    #             h_t = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)
    #         else:
    #             h_t = h0

    #         memory = []

    #         for t in range(T):
    #             x_t = x[:, t, :]  # [B, input_size]

    #             # Compute gates
    #             gates = (
    #                 torch.nn.functional.linear(x_t, self.W_ih, self.b_ih)
    #                 + torch.nn.functional.linear(h_t, self.W_hh, self.b_hh)
    #             )
    #             r, z, n = gates.chunk(3, dim=1)

    #             r = torch.sigmoid(r)
    #             z = torch.sigmoid(z)
    #             n = torch.tanh(n + r * (torch.nn.functional.linear(h_t, self.W_hh[2*self.hidden_size:], None)))

    #             h_t = (1 - z) * n + z * h_t
    #             memory.append(h_t.unsqueeze(1))

    #         memory = torch.cat(memory, dim=1)  # [B, T, hidden_size]
    #         return memory, h_t
    class ManualGRU(torch.nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size

            # Gate order: r, z, n
            self.W_ih = torch.nn.Parameter(torch.randn(3 * hidden_size, input_size))
            self.W_hh = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
            self.b_ih = torch.nn.Parameter(torch.randn(3 * hidden_size))
            self.b_hh = torch.nn.Parameter(torch.randn(3 * hidden_size))
        def sigmoid(self, x):
            return 1.0 / (1.0 + torch.exp(-x))
        def forward(self, x, h0=None):
            B, T, _ = x.shape
            if h0 is None:
                h_t = torch.zeros(B, self.hidden_size, dtype=x.dtype, device=x.device)
            else:
                h_t = h0

            memory = []

            # Split weights/biases by gate
            W_ir, W_iz, W_in = self.W_ih.chunk(3, 0)
            W_hr, W_hz, W_hn = self.W_hh.chunk(3, 0)
            b_ir, b_iz, b_in = self.b_ih.chunk(3, 0)
            b_hr, b_hz, b_hn = self.b_hh.chunk(3, 0)

            for t in range(T):
                x_t = x[:, t, :]  # [B, input_size]
                # gi = x_t @ self.W_ih.T + self.b_ih          # [3*H]
                # gh = h_t @ self.W_hh.T + self.b_hh          # [3*H]
                # H = self.hidden_size
                # print(x_t.shape, W_ir.shape,b_ir.shape)
                # r_t = torch.sigmoid(torch.nn.functional.linear(x_t, W_ir, b_ir) + torch.nn.functional.linear(h_t, W_hr, b_hr))
                r_t = self.sigmoid(x_t @ W_ir.T  + b_ir + h_t @ W_hr.T  + b_hr)
                # r_t = torch.sigmoid(gi[:,:H] + gh[:,:H])
                z_t = self.sigmoid(x_t @ W_iz.T  + b_iz + h_t @W_hz.T +b_hz)
                # z_t = torch.sigmoid(gi[:, H:2*H] + gh[:, H:2*H])
                # z_t = torch.sigmoid(torch.nn.functional.linear(x_t, W_iz, b_iz) + torch.nn.functional.linear(h_t, W_hz, b_hz))
                # n_t = torch.tanh(torch.nn.functional.linear(x_t, W_in, b_in) + r_t * (torch.nn.functional.linear(h_t, W_hn, b_hn)))
                n_t = torch.tanh(   x_t @ W_in.T  + b_in + r_t * (h_t @ W_hn.T + b_hn))
                # n_t = torch.tanh(gi[:, 2*H:] + r_t * gh[:, 2*H:])

                h_t = (1 - z_t) * n_t + z_t * h_t
                # print("h[:3]:", h_t[0,0,:3])
                memory.append(h_t.unsqueeze(1))

            memory = torch.cat(memory, dim=1)
            return memory, h_t
    B, T, C, F = 1, 3, 2, 7  # Example sizes
    input_size = C * F
    hidden_size = 8

    # Example input (same as before transpose in your case)
    x = torch.randn(B, C, T, F)
    x_flat = x.permute(0, 2, 1, 3).contiguous().view(B, T, -1)  # [B, T, C*F]

    gru = ManualGRU(input_size, hidden_size)
    memory, last = gru(x_flat)
    gru_ref = torch.nn.GRU(input_size, hidden_size, batch_first=True)

    # Copy parameters
    with torch.no_grad():
        gru_ref.weight_ih_l0.copy_(gru.W_ih)
        gru_ref.weight_hh_l0.copy_(gru.W_hh)
        gru_ref.bias_ih_l0.copy_(gru.b_ih)
        gru_ref.bias_hh_l0.copy_(gru.b_hh)

        out_ref, h_ref = gru_ref(x_flat)

        print("Diff (out):", (out_ref - memory).abs().max())
        print("Diff (h):", (h_ref.squeeze(0) - last).abs().max())

    # print("Memory:", memory)
    # print("Last hidden:", last)


# check_conv_weight_norm()
check_gru()
