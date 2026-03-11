import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import time

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
            torch.from_numpy(node_target[:,:9]),
            torch.from_numpy(triangle_target[:,:8])
        )

class LinearAttention(nn.Module):
    """
    线性注意力机制实现 (Linear Transformer / Performer style)
    复杂度: O(N * d^2) 而不是 O(N^2 * d)
    适用于长序列 (如 60k+ nodes)
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # 合并 Q, K, V 投影以简化代码，类似原生 MultiheadAttention
        self.in_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # 特征映射函数：ELU + 1 保证正值，这是线性注意力的关键
        # 也可以使用 ReLU，但 ELU+1 在实验中通常更稳定
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False):
        """
        query: [B, N_q, D]
        key:   [B, N_k, D]
        value: [B, N_k, D]
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # 1. 投影 Q, K, V
        qkv = self.in_proj(torch.cat([query, key, value], dim=-1)) # [B, N, 3D] (注意这里如果是 cross attention，长度可能不同，需分开处理)
        
        # 为了处理 Cross Attention (Q 和 K/V 长度不同)，我们分开投影
        q = self.in_proj.weight[:D, :] @ query.transpose(-1, -2) + self.in_proj.bias[:D] # [B, D, N_q]
        k = self.in_proj.weight[D:2*D, :] @ key.transpose(-1, -2) + self.in_proj.bias[D:2*D]       # [B, D, N_k]
        v = self.in_proj.weight[2*D:, :] @ value.transpose(-1, -2) + self.in_proj.bias[2*D:]       # [B, D, N_k]
        
        # 转回 [B, Heads, Seq_Len, Head_Dim]
        q = q.reshape(B, self.n_heads, self.head_dim, N_q).transpose(2, 3) # [B, H, N_q, h]
        k = k.reshape(B, self.n_heads, self.head_dim, N_k).transpose(2, 3) # [B, H, N_k, h]
        v = v.reshape(B, self.n_heads, self.head_dim, N_k).transpose(2, 3) # [B, H, N_k, h]

        # 2. 应用特征映射 (Feature Map) 替代 Softmax
        # 这一步将负值变为正值，使得我们可以交换矩阵乘法顺序
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # 3. 线性注意力核心计算
        # 标准注意力: Softmax(QK^T)V
        # 线性注意力: (Q * (K^T * V)) / (Q * (K^T * 1))
        
        # 计算 KV 态 (State): [B, H, head_dim, head_dim]
        # 对序列维度 N_k 进行求和压缩
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v) 
        
        # 计算归一化分母态: [B, H, head_dim, 1]
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(dim=2)) + 1e-6)
        
        # 计算输出: [B, H, N_q, head_dim]
        out = torch.einsum("bhdm,bhn->bhnd", kv, q)
        
        # 应用归一化
        out = out * z.unsqueeze(-1)
        
        # 4. 合并头并投影输出
        out = out.transpose(2, 3).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out, None # 返回 None 作为 attn_weights 占位符，因为线性注意力没有显式的 N*N 权重矩阵

class LinearAttention(nn.Module):
    """
    通用线性注意力 (支持 Self 和 Cross Attention)
    允许 query_len != key_len == value_len
    复杂度: O((N_q + N_k) * d^2)
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # 独立的投影层，支持 Q, K, V 维度独立变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        # 特征映射: ELU + 1 保证正值，用于线性化 Softmax
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, query, key, value):
        """
        query: [B, N_q, D]
        key:   [B, N_k, D]
        value: [B, N_k, D]
        注意: N_q 可以不等于 N_k
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # 1. 线性投影
        q = self.q_proj(query)  # [B, N_q, D]
        k = self.k_proj(key)    # [B, N_k, D]
        v = self.v_proj(value)  # [B, N_k, D]
        
        # 2. 多头拆分: [B, N, D] -> [B, Heads, N, Head_Dim]
        q = q.reshape(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. 特征映射 (Kernel Feature Map)
        q = self.feature_map(q)
        k = self.feature_map(k)
        # v 不需要特征映射，直接参与加权求和
        
        # 4. 线性注意力核心计算
        # 目标: 计算 (Softmax(QK^T) @ V) 的线性近似
        # 公式: (phi(Q) @ (phi(K)^T @ V)) / (phi(Q) @ (phi(K)^T @ 1))
        
        # Step 4.1: 计算全局状态 KV (Global State)
        # k: [B, H, N_k, h], v: [B, H, N_k, h]
        # kv: [B, H, h, h]  <-- 这个矩阵大小与序列长度 N_k 无关!
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)
        
        # Step 4.2: 计算归一化分母
        # k_sum: [B, H, h] (对 N_k 维度求和)
        k_sum = k.sum(dim=2)
        # z: [B, H, N_q] = q @ k_sum
        z = torch.einsum("bhnd,bhd->bhn", q, k_sum)
        # 避免除零
        z = 1.0 / (z + 1e-6)
        
        # Step 4.3: 计算输出
        # out: [B, H, N_q, h] = (q @ kv) * z
        # 先算 q @ kv: [B, H, N_q, h] @ [B, H, h, h] -> [B, H, N_q, h]
        out = torch.einsum("bhnd,bhdm->bhnm", q, kv) # 注意 einsum 下标对应
        # 修正 einsum: 
        # q: [B, H, N_q, h] (bhnd)
        # kv: [B, H, h, h] (bhdm) -> 这里的 d 是 head_dim, m 也是 head_dim
        # 结果应该是 [B, H, N_q, m] (即 head_dim)
        # 正确的 einsum: bhnd, bhdm -> bhnm
        out = torch.einsum("bhnd,bhdm->bhnm", q, kv)
        
        # 应用归一化
        out = out * z.unsqueeze(-1)
        
        # 5. 合并多头并输出投影
        # [B, H, N_q, h] -> [B, N_q, H*h] -> [B, N_q, D]
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        return out

class LinearEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Encoder 内部是 Self-Attention (Q=K=V)
        self.self_attn = LinearAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu
        
    def forward(self, src):
        # Self Attention
        attn_out = self.self_attn(src, src, src)
        src = src + self.dropout(attn_out)
        src = self.norm1(src)
        
        # FFN
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + ff_out
        src = self.norm2(src)
        return src

class FvcomTransformer(nn.Module):
    def __init__(self, node=60882, triangle=115443, node_var=13,
                 triangle_var=17, embed_dim=256, n_heads=8, num_transformer_layers=6,
                 dropout=0.1):
        super().__init__()
        
        # 1. 输入嵌入
        self.node_embedding_layer = nn.Linear(node_var, embed_dim)
        self.triangle_embedding_layer = nn.Linear(triangle_var, embed_dim)

        # 2. 全局编码器 (Self-Attention on Concatenated Sequence)
        encoder_layers = [
            LinearEncoderLayer(d_model=embed_dim, nhead=n_heads, 
                               dim_feedforward=embed_dim*4, dropout=dropout)
            for _ in range(num_transformer_layers)
        ]
        self.global_encoder = nn.ModuleList(encoder_layers)
        
        # 3. 解码器交叉注意力 (Cross-Attention)
        # Q 是分离的 (Node/Tri), KV 是全局的
        # 这里复用同一个 LinearAttention 类，因为它支持 Cross Attention
        self.cross_att_node = LinearAttention(embed_dim, n_heads, dropout=dropout)
        self.cross_att_tri = LinearAttention(embed_dim, n_heads, dropout=dropout)
        
        # 4. 输出头
        self.node_out_head = nn.Linear(embed_dim, max(1, 9)) 
        self.elem_out_head = nn.Linear(embed_dim, max(1, 8))

    def forward(self, node_in, triangle_in):
        # 处理输入维度 (兼容 [B, 1, N, D] 或 [B, N, D])
        # if node_in.dim() == 3 and node_in.size(1) == 1:
        #     node_in = node_in.squeeze(1)
        # if triangle_in.dim() == 3 and triangle_in.size(1) == 1:
        #     triangle_in = triangle_in.squeeze(1)
        node_in = node_in.squeeze(0)
        triangle_in = triangle_in.squeeze(0)
            
        b = node_in.size(0)
        
        # --- Step 1: Embedding ---
        print(node_in.shape)
        print(triangle_in.shape)
        node_feats = self.node_embedding_layer(node_in)      # [B, N_node, D]
        triangle_feats = self.triangle_embedding_layer(triangle_in) # [B, N_tri, D]

        # --- Step 2: Concatenate for Global Context ---
        # 拼接: [B, N_node + N_tri, D] (~17w)
        combined_input = torch.cat([node_feats, triangle_feats], dim=1)
        n_node = node_feats.size(1)
        
        # --- Step 3: Global Encoder (Self-Attention) ---
        # 17w 个点互相做注意力，更新特征
        global_context = combined_input
        for layer in self.global_encoder:
            global_context = layer(global_context)
        # global_context shape: [B, N_total, D]
        
        # --- Step 4: Decoder (Cross-Attention) ---
        # 策略: 
        # Node Branch: Q=node_feats (原始或嵌入后的), KV=global_context
        # Tri Branch:  Q=tri_feats, KV=global_context
        # 这样每个节点/单元都能从全局上下文中“检索”相关信息
        
        # 使用原始的嵌入特征作为 Query (保留输入细节)，也可以使用 global_context 切片作为 Query
        # 这里选择使用 global_context 切分后作为 Query，因为经过 Encoder 后 Query 本身也包含了局部上下文
        # 如果你想用纯原始输入作为 Query，可以用 node_feats / triangle_feats 替换下面的 query_node/query_tri
        
        query_node = global_context[:, :n_node, :]       # [B, N_node, D]
        query_tri = global_context[:, n_node:, :]        # [B, N_tri, D]
        
        # Key & Value 都是完整的全局上下文
        key_global = global_context
        value_global = global_context
        
        # Cross Attention: Node
        # Q: [B, N_node, D], K: [B, N_total, D], V: [B, N_total, D]
        # Output: [B, N_node, D]
        node_out_feats = self.cross_att_node(query_node, key_global, value_global)
        
        # Cross Attention: Triangle
        # Q: [B, N_tri, D], K: [B, N_total, D], V: [B, N_total, D]
        # Output: [B, N_tri, D]
        tri_out_feats = self.cross_att_tri(query_tri, key_global, value_global)
        
        # --- Step 5: Output Heads ---
        next_node_state = self.node_out_head(node_out_feats) 
        next_elem_state = self.elem_out_head(tri_out_feats)
        
        return next_node_state, next_elem_state

    def predict(self, node_input_data, triangle_input_data, checkpoint_name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_name, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.to(device)
        self.eval()
        
        node_input_data = node_input_data.to(device)
        triangle_input_data = triangle_input_data.to(device)

        with torch.no_grad():
            output = self(node_input_data, triangle_input_data)
        return output


def run_test():
        print("="*30)
        print("FVCOM Linear Transformer Test")
        print("="*30)

        # 1. 配置参数 (使用缩小版数据测试流程，避免初次运行显存不足)
        # 真实场景: node=60882, triangle=115443
        # 测试场景: 我们先用小数据跑通，再尝试大数据
        use_full_scale = True  # 设置为 True 直接测试 6w+ 节点 (需要大显存 GPU)
        
        if use_full_scale:
            N_NODE = 60882
            N_TRI = 115443
            BATCH_SIZE = 1
            print(f"[模式] 全量测试 (Nodes: {N_NODE}, Tris: {N_TRI})")
        else:
            N_NODE = 5000   # 缩小版测试
            N_TRI = 9000
            BATCH_SIZE = 2
            print(f"[模式] 快速流程测试 (Nodes: {N_NODE}, Tris: {N_TRI})")

        NODE_VAR = 13
        TRI_VAR = 17
        EMBED_DIM = 256
        N_HEADS = 8
        NUM_LAYERS = 6      # 测试时减少层数以加快速度
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"设备: {DEVICE}")
        if DEVICE.type == 'cpu':
            print("警告: 正在使用 CPU 运行。对于线性注意力，CPU 速度会非常慢，建议使用 CUDA。")

        # 2. 模拟输入数据
        # 形状: [Batch, Seq_Len, Features]
        # 最后两列假设为坐标 (x, y)，前面为物理量
        print("\n正在生成模拟数据...")
        node_in = torch.randn(BATCH_SIZE, N_NODE, NODE_VAR).to(DEVICE)
        tri_in = torch.randn(BATCH_SIZE, N_TRI, TRI_VAR).to(DEVICE)
        
        # 模拟坐标 (x, y) 填入最后两列 (0~1 之间)
        node_in[:, :, -2:] = torch.rand(BATCH_SIZE, N_NODE, 2).to(DEVICE)
        tri_in[:, :, -2:] = torch.rand(BATCH_SIZE, N_TRI, 2).to(DEVICE)

        # 3. 初始化模型
        print("正在初始化模型...")
        model = FvcomTransformer(
            node=N_NODE,
            triangle=N_TRI,
            node_var=NODE_VAR,
            triangle_var=TRI_VAR,
            embed_dim=EMBED_DIM,
            n_heads=N_HEADS,
            num_transformer_layers=NUM_LAYERS,
            dropout=0.1
        ).to(DEVICE)

        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params / 1e6:.2f} M")

        # 4. 预热 (Warmup)
        print("\n正在预热 (Warmup)...")
        model.eval()
        with torch.no_grad():
            _ = model(node_in, tri_in)
        
        # 清理缓存
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        # 5. 正式测试
        print("\n开始正式前向传播计时...")
        start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            try:
                node_out, tri_out = model(node_in, tri_in)
            except RuntimeError as e:
                print(f"\n[错误] 运行时错误: {e}")
                if "out of memory" in str(e):
                    print("建议: 显存不足。请减小 N_NODE/N_TRI 或 batch_size，或使用更小的 embed_dim。")
                return

        end_time = time.time()
        elapsed = end_time - start_time

        # 6. 结果验证
        print("\n" + "="*30)
        print("测试结果:")
        print("="*30)
        print(f"输入 Node 形状: {node_in.shape}")
        print(f"输入 Tri  形状: {tri_in.shape}")
        print(f"输出 Node 形状: {node_out.shape} (期望: [{BATCH_SIZE}, {N_NODE}, {9}])")
        print(f"输出 Tri  形状: {tri_out.shape} (期望: [{BATCH_SIZE}, {N_TRI}, {8}])")
        print(f"耗时: {elapsed:.4f} 秒")
        print(f"吞吐量: {(N_NODE + N_TRI) / elapsed:.2f} tokens/sec")
        
        if DEVICE.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated(DEVICE) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(DEVICE) / 1024**3
            print(f"显存占用 (Allocated): {mem_allocated:.2f} GB")
            print(f"显存占用 (Reserved):  {mem_reserved:.2f} GB")

        print("\n[成功] 流程测试通过！线性注意力机制运行正常。")

if __name__ == "__main__":
    run_test()