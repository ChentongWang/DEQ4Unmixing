import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import random
import torchvision.transforms as transforms
import os
from einops import rearrange
import scipy.sparse as sp
from skimage import graph

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Random Seed
seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose Dataset
data_name = 'samson'

if data_name == 'samson':
    image_file = r'./data/samson_dataset.mat'
    superpixel_file = r'./superpixel/samson_superpixel_labels.mat'
    L, R, H, W = 156, 3, 95, 95
    eta, Iter = 0.01, 10
    learning_rate, weight_decay = 0.01, 1e-5
    special_LR, special_wd = 0.007, 1e-5
    alpha = 0.2
    gamma = 1
    m, lam = 5, 0.1
    EPOCH = 300

elif data_name == 'muffle':
    image_file = r'./data/muffle_dataset.mat'
    superpixel_file = r'./superpixel/muffle_superpixel_labels.mat'
    L, R, H, W = 64, 5, 90, 130
    eta, Iter = 0.01, 10
    learning_rate, weight_decay = 0.0065, 1e-4
    special_LR, special_wd = 0.01, 0
    alpha = 4
    gamma = 3
    m, lam = 5, 0.1
    EPOCH = 300

else:
    raise ValueError("Unknown dataset")

# Load DATA
data = sio.loadmat(image_file)

abundance_GT = torch.from_numpy(data["A"])  # true abundance: R*N
original_HSI = torch.from_numpy(data["Y"])  # observed image: L*N
VCA_endmember = torch.from_numpy(data["M1"]).float()  # endmembers extracted by VCA: L*R
GT_endmember = torch.from_numpy(data["M"])  # true endmembers: L*R
abundance_FCLS = torch.from_numpy(data["A1"]).float()  # abundances estimated by FCLSU: R*N

band_Number = original_HSI.shape[0]  # L
endmember_number, pixel_number = abundance_GT.shape  # P, (col*col)

original_HSI = torch.reshape(original_HSI, (band_Number, H, W))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, H, W))
abundance_FCLS = torch.reshape(abundance_FCLS, (endmember_number, H, W))

batch_size = 1
drop_out = 0.


# calculate SAD of endmember
def SAD_distance(src, ref):
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim


# -----------------------------------------------Super-Pixel Prox-------------------------------------------------------
# loading the superpixel labels
label_data = sio.loadmat(superpixel_file)
superpixel_labels = label_data['labels']
labels_flat_tensor = torch.from_numpy(superpixel_labels).long().flatten().to(device)


def build_pool_matrix(labels_flat):
    labels_np = labels_flat.cpu().numpy()
    P = len(labels_np)
    S = labels_np.max() + 1

    rows = labels_np
    cols = np.arange(P)
    data = np.ones(P, dtype=np.float32)

    counts = np.bincount(rows, minlength=S)
    counts_inv = 1.0 / np.maximum(counts, 1e-6)
    data_mean = data * counts_inv[rows]

    indices = torch.from_numpy(np.vstack((rows, cols))).long()
    values = torch.from_numpy(data_mean).float()

    pool_mat = torch.sparse_coo_tensor(indices, values, torch.Size([S, P]))
    return pool_mat.coalesce()


pool_matrix_sparse = build_pool_matrix(labels_flat_tensor).to(device)


def build_normalized_adjacency(labels):
    num_nodes = labels.max() + 1
    rag = graph.RAG(labels)

    if not rag.edges():
        adj = sp.eye(num_nodes, dtype=np.float32)
    else:
        edges = np.array(list(rag.edges()))
        rows = edges[:, 0]
        cols = edges[:, 1]
        data = np.ones(len(rows), dtype=np.float32)
        adj = sp.coo_matrix((data, (rows, cols)),
                            shape=(num_nodes, num_nodes), dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj_hat = adj + sp.eye(num_nodes, dtype=np.float32)
    rowsum = np.array(adj_hat.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    d_inv_sqrt = sp.diags(r_inv_sqrt)
    normalized_adj = d_inv_sqrt.dot(adj_hat).dot(d_inv_sqrt)
    coo = normalized_adj.tocoo()
    indices = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    values = torch.from_numpy(coo.data).float()
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


adj_matrix_sparse = build_normalized_adjacency(superpixel_labels).to(device)


# Superpixel_Projection
def superpixel_projection(abundance_map, labels_flat, pool_mat, H, W):
    B, R, _, _ = abundance_map.shape
    abundance_flat = rearrange(abundance_map, 'b r h w -> (b h w) r')
    mean_abundances_by_sp = torch.sparse.mm(pool_mat, abundance_flat)
    projected_flat = mean_abundances_by_sp[labels_flat]
    projected_map = rearrange(projected_flat, '(b h w) r -> b r h w', b=B, h=H, w=W)
    return projected_map


# ----------------------------------------------------------------------------------------------------------------------
# Core Model [IMPLEMENTATION HIDDEN FOR REVIEW]
class DEQModel(nn.Module):
    """
    The core deep equilibrium architecture including the learnable gradient module (GRAD_A),
    superpixel-guided proximal operator, and fixed-point iteration logic is withheld.
    """

    def __init__(self, eta, L, R, H, W, max_iter, adj_matrix, labels_flat, pool_mat):
        super(DEQModel, self).__init__()

        self.L = L
        self.R = R
        # W_M is exposed to allow external initialization and optimizer parameter splitting
        self.W_M = nn.Parameter(torch.randn(self.R, self.L))

        # [CORE MODULES HIDDEN]
        # self.grad_model_A = ...
        # self.prox_net = ...

    @staticmethod
    def init_weights(m):
        # [INITIALIZATION HIDDEN]
        pass

    def forward(self, Y, A_0):
        # [DEQ SOLVER AND FORWARD PASS HIDDEN]
        # return estimated endmembers (M_result) and abundances (A_result)
        raise NotImplementedError("Core forward pass is hidden.")

    def backward(self, grad_output_M, grad_output_A):
        # [IMPLICIT DIFFERENTIATION HIDDEN]
        pass


# ----------------------------------------------------------------------------------------------------------------------
# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


# MSE loss of reconstruction
MSE = torch.nn.MSELoss(size_average=True)


# Reconstruct the HyperSpectral Image
def reconstruct_y(M, A):
    A_flat = A.squeeze(0).view(A.size(1), -1)
    Y_rec_flat = torch.matmul(M, A_flat)
    Y_rec = Y_rec_flat.view(1, M.size(0), A.size(2), A.size(3))
    return Y_rec


# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, H, W)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# ----------------------------------------------------------------------------------------------------------------------
# load data
class load_data(torch.utils.data.Dataset):
    def __init__(self, img, gt, transform=None):
        self.img = img.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.gt

    def __len__(self):
        return 1


train_dataset = load_data(
    img=original_HSI, gt=abundance_GT, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)

# Initializing the Net
net = DEQModel(eta=eta, L=L, R=R, H=H, W=W, max_iter=Iter, adj_matrix=adj_matrix_sparse,
               labels_flat=labels_flat_tensor, pool_mat=pool_matrix_sparse).to(device)

# weight init
net.apply(net.init_weights)

# Separate the parameter W_M
w_m_params = [net.W_M]
w_m_ids = {id(p) for p in w_m_params}
other_params = []

for p in net.parameters():
    if id(p) not in w_m_ids:
        other_params.append(p)

# Optimizer
optimizer = torch.optim.AdamW([
    {'params': w_m_params, 'lr': special_LR, 'weight_decay': special_wd},
    {'params': other_params, 'lr': learning_rate, 'weight_decay': weight_decay}
])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.8)

# Initialization of M_0
with torch.no_grad():
    vca_endmember_T = torch.tensor(VCA_endmember.T, dtype=torch.float32, device=device)
    vca_endmember_T = vca_endmember_T.clamp(min=1e-5)
    net.W_M.copy_(vca_endmember_T)

# Initialization of A_0
A_0 = abundance_FCLS.unsqueeze(0).cuda()

train_losses = []
abundance_losses = []
mse_losses = []

'''Train the model'''
for epoch in range(EPOCH):
    for i, (x, y) in enumerate(train_loader):
        scheduler.step()

        print(f'-------------{epoch}------------')

        x = x.cuda()

        # NOTE: Forward pass logic will raise NotImplementedError with the hidden model
        M_result, A_result = net(x, A_0)

        reconstruction_result = reconstruct_y(M_result, A_result)

        abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

        MSELoss = MSE(x, reconstruction_result)

        ALoss = abundanceLoss
        BLoss = MSELoss

        total_loss = ALoss + (alpha * BLoss)

        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1, norm_type=1)
        optimizer.zero_grad()

        total_loss.backward()

        optimizer.step()
        with torch.no_grad():
            net.W_M.data = net.W_M.data.clamp(min=1e-5)

        train_losses.append(total_loss.item())
        abundance_losses.append(ALoss.item())
        mse_losses.append(BLoss.item())

        if epoch % 10 == 0:
            print(
                "Epoch:",
                epoch,
                "| Abundanceloss: %.4f" % ALoss.cpu().data.numpy(),
                "| MSEloss: %.4f" % (alpha * BLoss).cpu().data.numpy(),
                "| total_loss: %.4f" % total_loss.cpu().data.numpy(),
            )

net.eval()

M_res, A_res = net(x, A_0)
M0 = M_res.cpu().detach().numpy()
en_abundance, abundance_GT = norm_abundance_GT(A_res, abundance_GT)

# Computing aRMSE
rmse_list = []
for i in range(endmember_number):
    rmse = np.sqrt(((en_abundance[i, :, :] - abundance_GT[i, :, :]) ** 2).sum() / (H * W))
    rmse_list.append(rmse)

aRMSE = np.mean(rmse_list)
print('aRMSE:', aRMSE)

# Computing mSAD
sad_list = []
for i in range(endmember_number):
    sad = SAD_distance(M0[:, i], GT_endmember[:, i])
    sad_list.append(sad)

mSAD = np.mean(sad_list)
print('mSAD:', mSAD)

# Save the results as '.mat'
sio.savemat(f'results/{data_name}_results.mat', {'A0': en_abundance, 'M0': M0})