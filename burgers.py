import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import os
import datetime
from tqdm import trange
import matplotlib.pyplot as plt
import time

torch.manual_seed(42)
np.random.seed(42)


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_dtype(torch.float32)
    DTYPE = torch.float32
    NP_DTYPE = np.float32
    torch.backends.cudnn.benchmark = True
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('mps')
    torch.set_default_dtype(torch.float32)
    DTYPE = torch.float32
    NP_DTYPE = np.float32
    print("Using mps")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")


class MLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.layers[i].weight)
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)

class ModifiedMLP(nn.Module):
    def __init__(self, layers, activation=nn.Tanh()):
        super(ModifiedMLP, self).__init__()
        self.activation = activation

        self.U1 = nn.Linear(layers[0], layers[1])
        self.U2 = nn.Linear(layers[0], layers[1])

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))

        for layer in [self.U1, self.U2] + list(self.layers):
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        U = self.activation(self.U1(x))
        V = self.activation(self.U2(x))

        inputs = x
        for i, layer in enumerate(self.layers[:-1]):
            outputs = self.activation(layer(inputs))
            inputs = outputs * U + (1 - outputs) * V

        return self.layers[-1](inputs)

class SimpleFourierNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        self.B = nn.Parameter(torch.randn(input_dim, hidden_dim) * 10.0, requires_grad=False)

        layers = [nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_proj = torch.matmul(x, self.B)
        x_fourier = torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return self.net(x_fourier)

class TrainableBasisFunction(nn.Module):
    """A trainable basis function"""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )


        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class HBPP(nn.Module):
    """HBPP with learnable basis functions for the Burgers equation."""
    def __init__(self, n_basis, branch_layers, trunk_layers, x_sensors):
        super().__init__()

        # Trainable basis functions
        self.basis_functions = nn.ModuleList([
            TrainableBasisFunction() for _ in range(n_basis)
        ])

        self.register_buffer('x_sensors', torch.tensor(x_sensors, dtype=DTYPE))
        self.n_basis = n_basis
        self.m = len(x_sensors)
        self.hidden_dim = branch_layers[-1]

        # Branch takes ALL basis function values at sensors
        # So input dimension should be n_basis * m
        branch_input_dim = n_basis * self.m
        branch_layers_full = [branch_input_dim] + branch_layers[1:]


        self.branch_net = ModifiedMLP(branch_layers_full)
        self.trunk_net = SimpleFourierNet(2, trunk_layers[1], self.hidden_dim, len(trunk_layers)-1)


        self.scaling_net = MLP([2 * self.hidden_dim, 64, 2])
        self.trans_net = MLP([self.hidden_dim+self.n_basis, 64, self.hidden_dim])

        self.out_proj = nn.Linear(self.hidden_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        self.to(device)

    def get_basis_at_sensors(self):
        """Evaluate all basis functions at sensor points"""
        x_sensors_input = self.x_sensors.unsqueeze(1)  
        basis_vals = []

        for basis_func in self.basis_functions:
            vals = basis_func(x_sensors_input).squeeze() 
            basis_vals.append(vals)

        # (n_basis, m) -> (n_basis * m,)
        basis_tensor = torch.stack(basis_vals).flatten()
        return basis_tensor

    def forward(self, x, t):
        """Forward pass for a single IC (represented by basis functions)"""
        batch_size = x.shape[0]


        if not x.requires_grad:
            x = x.requires_grad_(True)
        if not t.requires_grad:
            t = t.requires_grad_(True)

        # Get basis representation of the IC (same for all points in batch)
        basis_at_sensors = self.get_basis_at_sensors()  # (n_basis * m,)

        # Branch network - same output for all points since IC is fixed
        branch_out = self.branch_net(basis_at_sensors.unsqueeze(0))  # (1, hidden_dim)
        B = branch_out.expand(batch_size, -1)  # (batch_size, hidden_dim)

        # Trunk network - different for each (x,t) point
        coords = torch.stack([x, t], dim=1)  # (batch_size, 2)
        T = self.trunk_net(coords)  # (batch_size, hidden_dim)


        # Query from trunk, Key from branch
        query = T.unsqueeze(1)  # (batch, 1, hidden_dim)
        key = B.unsqueeze(1)    # (batch, 1, hidden_dim)

        # Scaling network
        concat_features = torch.cat([B, T], dim=-1)
        scaling_output = self.scaling_net(concat_features)
        mean = torch.tanh(scaling_output[:, 0:1]) * 0.1
        var = torch.sigmoid(scaling_output[:, 1:2]) * 0.8 + 0.6

        # Scale coordinates
        coords_scaled = torch.stack([
            (coords[:, 0] - mean.squeeze()) / var.squeeze(),
            (coords[:, 1] - mean.squeeze()) / var.squeeze()
        ], dim=1)

        # Value computation using basis functions
        # Evaluate basis at scaled x coordinates
        x_scaled = coords_scaled[:, 0].unsqueeze(1)  # (batch, 1)
        basis_at_x = []
        for basis_func in self.basis_functions:
            vals = basis_func(x_scaled).squeeze()  # (batch,)
            basis_at_x.append(vals)

        basis_at_x = torch.stack(basis_at_x, dim=1)  # (batch, n_basis)

        # Transform to value
        value_input = torch.cat([T, basis_at_x], dim=1)
        value_transformed = self.trans_net(value_input) 
        value = value_transformed.unsqueeze(1)  # (batch, 1, hidden_dim)


        scale_factor = math.sqrt(self.hidden_dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / scale_factor
        attn_weights = torch.sigmoid(attn_scores)
        context = attn_weights* value


        output = self.out_proj(context.squeeze(1))
        output = torch.tanh(output + self.bias) * 1.5

        return output

    def compute_residual(self, x, t):
        """Compute PDE residual"""
        u = self.forward(x, t)

        # Gradients
        u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
        u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]

        # Burgers equation
        nu = 0.01 / np.pi
        residual = u_t + u.squeeze() * u_x - nu * u_xx

        return residual

    def loss_fn(self, x_ic, x_bc, t_bc, x_r, t_r, iteration):
        """Compute loss for single IC: -sin(pix)"""
        # IC loss
        t_ic = torch.zeros_like(x_ic)
        u_ic_pred = self.forward(x_ic, t_ic)
        u_ic_true = -torch.sin(np.pi * x_ic).unsqueeze(1)
        loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)

        # BC loss
        x_left = torch.full_like(t_bc, -1.0)
        x_right = torch.full_like(t_bc, 1.0)
        u_left = self.forward(x_left, t_bc)
        u_right = self.forward(x_right, t_bc)
        loss_bc = torch.mean(u_left**2) + torch.mean(u_right**2)

        # Physics loss
        res = self.compute_residual(x_r, t_r)
        loss_res = torch.mean(res**2)

        # Adaptive weighting
        w_physics = 0.01 if iteration < 1000 else (0.1 if iteration < 3000 else 1.0)

        total = loss_ic + loss_bc + w_physics * loss_res

        return total, loss_ic.item(), loss_bc.item(), loss_res.item()

def train_model(model, n_iter, batch_size, lr, output_folder):
    """Train the model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)

    log_file = open(os.path.join(output_folder, 'training_log.txt'), 'w')
    losses = {'total': [], 'ic': [], 'bc': [], 'res': []}

    pbar = trange(n_iter)
    best_loss = float('inf')
    start_time = time.time()

    for it in pbar:
        # Sample points
        x_ic = torch.rand(batch_size, device=device) * 2 - 1  # [-1, 1]
        t_bc = torch.rand(batch_size, device=device)  # [0, 1]
        x_r = torch.rand(batch_size, device=device) * 2 - 1
        t_r = torch.rand(batch_size, device=device)


        x_r.requires_grad_(True)
        t_r.requires_grad_(True)

        # Compute loss
        optimizer.zero_grad()
        loss, l_ic, l_bc, l_res = model.loss_fn(x_ic, x_ic, t_bc, x_r, t_r, it)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if it % 100 == 0:
            elapsed = time.time() - start_time
            it_per_sec = (it + 1) / elapsed

            losses['total'].append(loss.item())
            losses['ic'].append(l_ic)
            losses['bc'].append(l_bc)
            losses['res'].append(l_res)

            log_str = f"It: {it}, Loss: {loss.item():.2e}, IC: {l_ic:.2e}, BC: {l_bc:.2e}, Res: {l_res:.2e}, It/s: {it_per_sec:.1f}"
            pbar.set_description(log_str)
            log_file.write(log_str + '\n')

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), os.path.join(output_folder, 'best_model.pth'))

    log_file.close()

    # Plot training
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.semilogy(losses['total'])
    plt.title('Total Loss')
    plt.xlabel('Iteration (x100)')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.semilogy(losses['ic'], label='IC')
    plt.semilogy(losses['bc'], label='BC')
    plt.semilogy(losses['res'], label='Res')
    plt.legend()
    plt.title('Loss Components')
    plt.xlabel('Iteration (x100)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    # Plot basis functions
    x_plot = torch.linspace(-1, 1, 200, device=device).unsqueeze(1)
    with torch.no_grad():
        for i, basis in enumerate(model.basis_functions):
            if i < 10:  # Plot first 10
                y = basis(x_plot).cpu()
                plt.plot(x_plot.cpu(), y, alpha=0.7, label=f'Basis {i}')
    plt.xlabel('x')
    plt.ylabel('Basis value')
    plt.title('Learned Basis Functions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'training.png'))
    plt.close()

    print(f"\nTraining completed in {elapsed:.1f}s")
    print(f"Best loss: {best_loss:.2e}")

def evaluate_model(model, output_folder):
    """Evaluate the trained model"""
    model.load_state_dict(torch.load(os.path.join(output_folder, 'best_model.pth')))
    model.eval()

    # create grid
    nx, nt = 256, 101
    x = torch.linspace(-1, 1, nx, device=device)
    t = torch.linspace(0, 1, nt, device=device)
    X, T = torch.meshgrid(x, t, indexing='xy')

    # Predict solution
    print("Predicting solution...")
    with torch.no_grad():
        x_flat = X.flatten()
        t_flat = T.flatten()

        # Predict in batches
        u_pred = []
        batch_size = 10000
        for i in range(0, len(x_flat), batch_size):
            u_batch = model.forward(x_flat[i:i+batch_size], t_flat[i:i+batch_size])
            u_pred.append(u_batch.cpu())

        u_pred = torch.cat(u_pred).reshape(X.shape).numpy()

    # visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Basis functions representation
    axes[0, 0].set_title('Basis Functions at Sensors')
    basis_vals = model.get_basis_at_sensors().cpu().detach().numpy()
    basis_vals = basis_vals.reshape(model.n_basis, model.m)
    im = axes[0, 0].imshow(basis_vals, aspect='auto', cmap='RdBu')
    axes[0, 0].set_xlabel('Sensor index')
    axes[0, 0].set_ylabel('Basis function')
    plt.colorbar(im, ax=axes[0, 0])

    # solution heatmap
    im = axes[0, 1].imshow(u_pred, extent=[-1, 1, 0, 1], origin='lower',
                           aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    axes[0, 1].set_title('Predicted Solution')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('t')
    plt.colorbar(im, ax=axes[0, 1])

    # Initial condition check
    axes[0, 2].plot(x.cpu(), u_pred[0, :], 'b-', linewidth=2, label='Predicted')
    axes[0, 2].plot(x.cpu(), -np.sin(np.pi * x.cpu()), 'r--', linewidth=2, label='True IC')
    axes[0, 2].set_title('Initial Condition')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('u(x,0)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Time evolution
    time_indices = [0, 25, 50, 75, 100]
    for t_idx in time_indices:
        axes[1, 0].plot(x.cpu(), u_pred[t_idx, :], label=f't={t[t_idx]:.2f}')
    axes[1, 0].set_title('Solution Evolution')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('u(x,t)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Boundary values
    axes[1, 1].plot(t.cpu(), u_pred[:, 0], label='x=-1')
    axes[1, 1].plot(t.cpu(), u_pred[:, -1], label='x=1')
    axes[1, 1].set_title('Boundary Values')
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('u')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Check conservation
    l2_norms = []
    for i in range(nt):
        l2_norm = np.sqrt(np.mean(u_pred[i, :]**2))
        l2_norms.append(l2_norm)

    axes[1, 2].plot(t.cpu(), l2_norms)
    axes[1, 2].set_title('L2 Norm Evolution')
    axes[1, 2].set_xlabel('t')
    axes[1, 2].set_ylabel('||u||_2')
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'evaluation.png'))
    plt.close()

    # Check physics
    print("\nChecking physics residual...")
    with torch.enable_grad():
        n_test = 1000
        x_test = torch.rand(n_test, device=device) * 2 - 1
        t_test = torch.rand(n_test, device=device) * 0.8 + 0.1  # Avoid t=0

        x_test.requires_grad_(True)
        t_test.requires_grad_(True)

        res = model.compute_residual(x_test, t_test)
        res_np = res.detach().cpu().numpy()

        print(f"Residual stats:")
        print(f"  Mean: {np.mean(res_np):.2e}")
        print(f"  Std: {np.std(res_np):.2e}")
        print(f"  Max: {np.max(np.abs(res_np)):.2e}")

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_folder = f"burgers_basis_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Hyperparameters
    n_basis = 50  # Number of basis functions
    m = 50  # Number of sensors

    # Architecture
    branch_layers = [n_basis * m, 128, 128, 128]  
    trunk_layers = [2, 64, 128, 128]

    # Create model
    x_sensors = np.linspace(-1, 1, m)
    model = HBPP(n_basis, branch_layers, trunk_layers, x_sensors)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nTraining...")
    train_model(model, n_iter=10000, batch_size=2048, lr=1e-3, output_folder=output_folder)

    # Evaluate
    print("\nEvaluating...")
    evaluate_model(model, output_folder)

    print(f"\nDone! Results in {output_folder}/")