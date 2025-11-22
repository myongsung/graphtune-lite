import math
import numpy as np
import torch

# -----------------------------
# Random feature / kernel utils
# -----------------------------

def create_products_of_givens_rotations(dim: int, seed: int) -> torch.Tensor:
    """
    Structured orthogonal matrix via products of random Givens rotations.
    (Bugfix) legacy had cos used twice; second term must be sin.
    """
    nb_givens_rotations = dim * int(math.ceil(math.log(float(dim))))
    q = np.eye(dim, dim)
    np.random.seed(seed)
    for _ in range(nb_givens_rotations):
        random_angle = math.pi * np.random.uniform()
        random_indices = np.random.choice(dim, 2)
        index_i = min(random_indices[0], random_indices[1])
        index_j = max(random_indices[0], random_indices[1])
        slice_i = q[index_i]
        slice_j = q[index_j]

        # correct Givens rotation
        new_slice_i = math.cos(random_angle) * slice_i + math.sin(random_angle) * slice_j
        new_slice_j = -math.sin(random_angle) * slice_i + math.cos(random_angle) * slice_j

        q[index_i] = new_slice_i
        q[index_j] = new_slice_j
    return torch.tensor(q, dtype=torch.float32)


def create_random_matrix(
    m: int,
    d: int,
    seed: int = 0,
    scaling: int = 0,
    struct_mode: bool = False,
) -> torch.Tensor:
    """
    Create random projection matrix (m x d).
    (Patch) torch.qr -> torch.linalg.qr
    """
    nb_full_blocks = int(m / d)
    block_list = []
    current_seed = seed

    for _ in range(nb_full_blocks):
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
            q = q.t()
        block_list.append(q)
        current_seed += 1

    remaining_rows = m - nb_full_blocks * d
    if remaining_rows > 0:
        torch.manual_seed(current_seed)
        if struct_mode:
            q = create_products_of_givens_rotations(d, current_seed)
        else:
            unstructured_block = torch.randn((d, d))
            q, _ = torch.linalg.qr(unstructured_block, mode="reduced")
            q = q.t()
        block_list.append(q[0:remaining_rows])

    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    if scaling == 0:
        multiplier = torch.norm(torch.randn((m, d)), dim=1)
    elif scaling == 1:
        multiplier = torch.sqrt(torch.tensor(float(d))) * torch.ones(m)
    else:
        raise ValueError("Scaling must be one of {0, 1}.")

    return torch.matmul(torch.diag(multiplier), final_matrix)


def random_feature_map(
    data: torch.Tensor,
    is_query: bool,
    projection_matrix: torch.Tensor,
    numerical_stabilizer: float = 1e-6,
) -> torch.Tensor:
    # legacy logic 그대로
    data_normalizer = 1.0 / torch.sqrt(
        torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32, device=data.device))
    )
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(
        torch.tensor(projection_matrix.shape[0], dtype=torch.float32, device=data.device)
    )
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)

    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape) - 1) / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape) - 1)

    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3

    if is_query:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]
            )
            + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(
                data_dash
                - diag_data
                - torch.max(
                    torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t,
                    keepdim=True,
                )[0]
            )
            + numerical_stabilizer
        )
    return data_dash


def linear_kernel(
    x: torch.Tensor,
    node_vec1: torch.Tensor,
    node_vec2: torch.Tensor,
) -> torch.Tensor:
    # legacy logic 그대로
    node_vec1 = node_vec1.permute(1, 0, 2, 3)
    node_vec2 = node_vec2.permute(1, 0, 2, 3)
    x = x.permute(1, 0, 2, 3)

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)

    one_matrix = torch.ones([node_vec2.shape[0]], device=node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)

    out1 = out1.permute(1, 0, 2, 3)
    out2 = out2.permute(1, 0, 2).unsqueeze(-1)
    return out1 / out2
