import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os

class FvcomDataset(Dataset):
    def __init__(
        self,
        node_data_dir: str,
        triangle_data_dir: str,
        total_timesteps: int = 144 * 7,
        steps_per_file: int = 144,
        input_steps: int = 1,
        pred_step: int = 1
    ):
        self.node_data_dir = node_data_dir
        self.triangle_data_dir = triangle_data_dir
        self.steps_per_file = steps_per_file
        self.input_steps = input_steps
        self.pred_step = pred_step

        self.node_files = sorted([f for f in os.listdir(node_data_dir) if f.endswith('.npy')])
        self.triangle_files = sorted([f for f in os.listdir(triangle_data_dir) if f.endswith('.npy')])

        assert len(self.node_files) == len(self.triangle_files), "Number of node and triangle files must match!"
        
        # expected_total = len(self.node_files) * steps_per_file
        # if total_timesteps != expected_total:
        #     raise ValueError(
        #         f"total_timesteps={total_timesteps} does not match "
        #         f"len(files)*steps_per_file = {expected_total}"
        #     )

        self.total_timesteps = total_timesteps
        self.max_start_t = total_timesteps - input_steps - pred_step
        if self.max_start_t < 0:
            raise ValueError(
                f"input_steps ({input_steps}) + pred_step ({pred_step}) > total_timesteps ({total_timesteps})"
            )
        self.total_samples = self.max_start_t + 1

    def _global_to_local(self, global_t: int):
        file_idx = global_t // self.steps_per_file
        local_t = global_t % self.steps_per_file
        return file_idx, local_t

    def _load_sequence(self, data_dir: str, files: list, global_t: int, length: int):
        if global_t + length > self.total_timesteps:
            raise IndexError(f"Requested sequence [{global_t}, {global_t + length}) exceeds total_timesteps={self.total_timesteps}")

        chunks = []
        remaining = length
        current_t = global_t

        while remaining > 0:
            file_idx, local_t = self._global_to_local(current_t)
            if file_idx >= len(files):
                raise RuntimeError(f"File index {file_idx} out of range. Check total_timesteps.")

            available_in_file = self.steps_per_file - local_t
            take_steps = min(remaining, available_in_file)

            path = os.path.join(data_dir, files[file_idx])
            data = np.load(path)
            chunk = data[local_t : local_t + take_steps]
            chunks.append(chunk)

            remaining -= take_steps
            current_t += take_steps

        return np.concatenate(chunks, axis=0)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        t_start = idx
        t_end_input = t_start + self.input_steps
        t_target = t_end_input + self.pred_step - 1

        node_input = self._load_sequence(self.node_data_dir, self.node_files, t_start, self.input_steps)
        triangle_input = self._load_sequence(self.triangle_data_dir, self.triangle_files, t_start, self.input_steps)

        node_target = self._load_sequence(self.node_data_dir, self.node_files, t_target, 1).squeeze(0)
        triangle_target = self._load_sequence(self.triangle_data_dir, self.triangle_files, t_target, 1).squeeze(0)
        return (
            torch.from_numpy(node_input),
            torch.from_numpy(triangle_input)
        ), (
            torch.from_numpy(node_target[:,:-5]),
            torch.from_numpy(triangle_target[:,:-4])
        )

class FvcomTransformer(nn.Module):
    def __init__(self, node=60882, triangle=115443, node_var=13,
                 triangle_var=17, embed_dim=256, n_latents=512, n_heads=8, num_transformer_layers=6,
                 dropout=0.1):
        super().__init__()
        
        self.node_embedding_layer = nn.Linear(node_var, embed_dim)
        self.triangle_embedding_layer = nn.Linear(triangle_var, embed_dim)

        # self.node_embed = nn.Sequential(
        #     nn.Linear(node_in_dim + 2, embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     nn.GELU()
        # )
        # # 单元路径：输入维度 = 4个物理量 + 2个坐标(x,y)
        # self.elem_embed = nn.Sequential(
        #     nn.Linear(elem_in_dim + 2, embed_dim),
        #     nn.LayerNorm(embed_dim),
        #     nn.GELU()
        # )

        self.latents_layer = nn.Parameter(torch.randn(n_latents, embed_dim))
        
        self.cross_att_in = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
        self.latent_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, 
                                       dim_feedforward=embed_dim*4, batch_first=True),
            num_layers=num_transformer_layers
        )

        self.node_to_query = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        self.triangle_to_query = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )

        self.cross_att_out = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        
        self.node_out_head = nn.Linear(embed_dim, node_var[:,:-5])
        self.elem_out_head = nn.Linear(embed_dim, triangle_var[:,:-4])

        # self.node_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim // 2, node_var)
        # )
        
        # self.triangle_head = nn.Sequential(
        #     nn.LayerNorm(embed_dim),
        #     nn.Linear(embed_dim, embed_dim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim // 2, triangle_var)
        # )

    def forward(self, node_in, triangle_in):
        node_in = node_in.squeeze(0)
        triangle_in = triangle_in.squeeze(0)
        # print(node_in.shape)
        b = node_in.size(0)
        node = self.node_embedding_layer(node_in)
        triangle = self.triangle_embedding_layer(triangle_in)

        combined_input = torch.cat([node, triangle], dim=1)

        latents = self.latents_layer.unsqueeze(0).expand(b, -1, -1)
        
        latent_feats, _ = self.cross_att_in(query=latents, 
                                            key=combined_input, 
                                            value=combined_input)
        
        latent_feats = self.latent_transformer(latent_feats)
        
        node_q = self.node_to_query(node_in[:,:,-2:])
        elem_q = self.triangle_to_query(triangle_in[:,:,-2:])
        
        node_out_feats, _ = self.cross_att_out(query=node_q, 
                                               key=latent_feats, 
                                               value=latent_feats)
        elem_out_feats, _ = self.cross_att_out(query=elem_q, 
                                               key=latent_feats, 
                                               value=latent_feats)
        
        next_node_state = self.node_out_head(node_out_feats) 
        next_elem_state = self.elem_out_head(elem_out_feats)
        
        return next_node_state, next_elem_state
    def predict(self, node_input_data, triangle_input_data, checkpoint_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)

        self.load_state_dict(checkpoint['model_state_dict'])

        self.eval()
        with torch.no_grad():
            output = self(node_input_data, triangle_input_data)

        # prediction = output.squeeze(0).cpu().numpy()
        
        return output

def FvcomModel(node=60882, triangle=115443, node_var=13,
               triangle_var=17, embed_dim=256,
               n_latents=512, n_heads=8, num_transformer_layers=6,
               dropout=0.1):
    
    model = FvcomTransformer(node=node, triangle=triangle, node_var=node_var,
                             triangle_var=triangle_var, embed_dim=embed_dim,
                             n_latents=n_latents, n_heads=n_heads,
                             num_transformer_layers=num_transformer_layers,
                             dropout=dropout)
    
    
    
    return model


# model = FvcomTransformer()
# sample_input1 = torch.randn(1, 70000, 11)
# sample_input2 = torch.randn(1, 120000, 15)

# output = model(sample_input1, sample_input2)
# print(f"输入形状: {sample_input1.shape}")
# print(f"输入形状: {sample_input2.shape}")
# print(f"输出形状: {output[0].shape}")
# print(f"输出形状: {output[1].shape}")