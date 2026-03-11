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
    通用线性注意力模块 (Generalized Linear Attention)
    
    特性:
    1. 支持 Self-Attention (Q, K, V 长度相同)
    2. 支持 Cross-Attention (Q 长度 != K/V 长度)
    3. 时间复杂度: O((N_q + N_k) * d^2)，空间复杂度: O(N * d)
    4. 使用 ELU+1 作为特征映射函数 (Feature Map) 以近似 Softmax
    """
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        # 独立的投影层，允许 Q, K, V 来自不同的分布或长度
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        # 特征映射: phi(x) = ELU(x) + 1，保证输出为正，用于线性化注意力
        self.feature_map = lambda x: F.elu(x) + 1.0

    def forward(self, query, key, value, key_padding_mask=None):
        """
        参数:
            query: [Batch, N_q, Embed_Dim]
            key:   [Batch, N_k, Embed_Dim]
            value: [Batch, N_k, Embed_Dim]
            key_padding_mask: [Batch, N_k] (可选，布尔掩码，True表示忽略)
            
        返回:
            output: [Batch, N_q, Embed_Dim]
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # 1. 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 2. 多头变换: [B, N, D] -> [B, Heads, N, Head_Dim]
        q = q.reshape(B, N_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N_k, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用特征映射 (Feature Map)
        # Q 和 K 需要映射到正空间，V 不需要
        q = self.feature_map(q)
        k = self.feature_map(k)
        
        # 4. 处理 Padding (如果有)
        if key_padding_mask is not None:
            # mask shape: [B, N_k] -> [B, 1, 1, N_k]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # 将 K 和 V 中 padding 位置置为 0
            k = k.masked_fill(mask, 0.0)
            v = v.masked_fill(mask, 0.0)
            
        # 5. 线性注意力核心计算 (Linear Attention Mechanism)
        # 标准注意力: Softmax(QK^T)V
        # 线性注意力: (phi(Q) * (phi(K)^T * V)) / (phi(Q) * (phi(K)^T * 1))
        
        # Step 5.1: 计算全局状态矩阵 KV (Global State Matrix)
        # k: [B, H, N_k, h], v: [B, H, N_k, h]
        # kv: [B, H, h, h]  <-- 注意：这里消去了 N_k 维度！
        # 公式: sum_{n=1}^{N_k} (k_n^T * v_n)
        kv = torch.einsum("bhnd,bhnm->bhdm", k, v)
        
        # Step 5.2: 计算归一化分母 (Normalizer)
        # k_sum: [B, H, h] <-- 对 N_k 求和
        k_sum = k.sum(dim=2)
        
        # z: [B, H, N_q]
        # 公式: q_n * sum(k)
        z = torch.einsum("bhnd,bhd->bhn", q, k_sum)
        
        # 防止除零，添加小量 epsilon
        z = 1.0 / (z + 1e-6)
        
        # Step 5.3: 计算最终输出
        # out_raw: [B, H, N_q, h]
        # 公式: q_n * KV_matrix
        out_raw = torch.einsum("bhnd,bhdm->bhnm", q, kv)
        
        # 应用归一化
        out = out_raw * z.unsqueeze(-1)
        
        # 6. 合并多头并投影输出
        # [B, H, N_q, h] -> [B, N_q, H*h] -> [B, N_q, D]
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        
        # 可选：Dropout
        out = self.dropout(out)
        
        return out

class LinearTransformerBlock(nn.Module):
    """
    标准的 Transformer Encoder Block
    包含: Linear Self-Attention + MLP + LayerNorm + Residual
    """
    def __init__(self, d_model, n_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = LinearAttention(d_model, n_heads, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = F.gelu
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self Attention (Q=K=V=src)
        attn_output = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feed Forward Network
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        return src

class FvcomTransformer(nn.Module):
    """
    FVCOM 专用线性 Transformer 架构
    
    架构流程:
    1. Input Embedding: Node 和 Triangle 分别映射到 D 维。
    2. Concatenation: 拼接成 [B, N_node+N_tri, D] 全局序列。
    3. Global Encoder: N 层 Linear Self-Attention，实现 17w 点的全局信息交互。
    4. Split & Decode: 
       - Node Branch: Q=Node_Features, KV=Global_Context (Cross Attention)
       - Tri Branch:  Q=Tri_Features,  KV=Global_Context (Cross Attention)
    5. Output Heads: 预测下一时刻物理量。
    """
    def __init__(self, 
                 node=60882, 
                 triangle=115443, 
                 node_var=13, 
                 triangle_var=17,
                 node_out_features=9,   # 例如: 预测水位, u, v, temp, salt... (原13 - 5个静态/坐标)
                 triangle_out_features=8, # 例如: 预测某些单元属性 (原17 - 4个静态/坐标)
                 embed_dim=256,
                 n_heads=4,
                 num_transformer_layers=2, 
                 dim_feedforward=1024, 
                 dropout=0.1):
        super().__init__()
        
        self.node_num = node
        self.triangle_num = triangle
        self.embed_dim = embed_dim
        
        # --- 1. 输入嵌入层 ---
        self.node_embedding = nn.Linear(node_var, embed_dim)
        self.triangle_embedding = nn.Linear(triangle_var, embed_dim)
        
        # --- 2. 全局编码器 (Global Encoder) ---
        # 处理拼接后的 17w 序列
        encoder_layers = [
            LinearTransformerBlock(
                d_model=embed_dim, 
                n_heads=n_heads, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout
            ) for _ in range(num_transformer_layers)
        ]
        self.global_encoder = nn.ModuleList(encoder_layers)
        
        # --- 3. 解码器交叉注意力 (Decoder Cross-Attention) ---
        # 这里的 LinearAttention 专门用于 Cross Attention (Q_len != KV_len)
        self.node_decoder_attn = LinearAttention(embed_dim, n_heads, dropout=dropout)
        self.tri_decoder_attn = LinearAttention(embed_dim, n_heads, dropout=dropout)
        
        # --- 4. 输出预测头 ---
        self.node_head = nn.Linear(embed_dim, node_out_features)
        self.triangle_head = nn.Linear(embed_dim, triangle_out_features)
        
        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, node_data, triangle_data, node_mask=None, triangle_mask=None):
        """
        参数:
            node_data: [Batch, N_node, Node_Features]
            triangle_data: [Batch, N_tri, Tri_Features]
            node_mask: [Batch, N_node] (可选，布尔值，True为Padding)
            triangle_mask: [Batch, N_tri] (可选)
            
        返回:
            node_pred: [Batch, N_node, Node_Out_Features]
            tri_pred: [Batch, N_tri, Tri_Out_Features]
        """
        node_data = node_data.squeeze(1)
        triangle_data = triangle_data.squeeze(1)
        print(node_data.shape)
        print(triangle_data.shape)
        B = node_data.size(0)
        device = node_data.device
        
        # --- Step 1: Embedding ---
        # [B, N_node, D]
        node_emb = self.node_embedding(node_data)
        # [B, N_tri, D]
        tri_emb = self.triangle_embedding(triangle_data)
        
        # --- Step 2: Concatenation (构建全局场) ---
        # [B, N_total, D]  where N_total = N_node + N_tri
        global_input = torch.cat([node_emb, tri_emb], dim=1)
        
        # 构建全局 Mask (如果需要)
        global_mask = None
        if node_mask is not None or triangle_mask is not None:
            if node_mask is None: node_mask = torch.zeros(B, self.node_num, dtype=torch.bool, device=device)
            if triangle_mask is None: triangle_mask = torch.zeros(B, self.triangle_num, dtype=torch.bool, device=device)
            global_mask = torch.cat([node_mask, triangle_mask], dim=1)
        
        # --- Step 3: Global Encoding (Self-Attention) ---
        # 所有节点和单元在此阶段互相交换信息
        global_context = global_input
        for layer in self.global_encoder:
            global_context = layer(global_context, src_key_padding_mask=global_mask)
        
        # --- Step 4: Decoding with Cross-Attention ---
        # 策略：
        # Query: 使用全局上下文中对应的部分 (已经包含了局部交互信息)
        # Key/Value: 使用完整的全局上下文 (让每个点重新审视全场)
        
        # 分割全局上下文
        # global_context[:, :self.node_num, :] -> Node 部分
        # global_context[:, self.node_num:, :] -> Triangle 部分
        node_query = global_context[:, :self.node_num, :]
        tri_query = global_context[:, self.node_num:, :]
        
        # Key 和 Value 都是完整的全局场
        global_kv = global_context
        
        # Node 分支 Cross Attention
        # Q: [B, N_node, D], K: [B, N_total, D], V: [B, N_total, D]
        # 输出: [B, N_node, D]
        node_decoded = self.node_decoder_attn(
            query=node_query, 
            key=global_kv, 
            value=global_kv,
            key_padding_mask=global_mask
        )
        
        # Triangle 分支 Cross Attention
        # Q: [B, N_tri, D], K: [B, N_total, D], V: [B, N_total, D]
        # 输出: [B, N_tri, D]
        tri_decoded = self.tri_decoder_attn(
            query=tri_query, 
            key=global_kv, 
            value=global_kv,
            key_padding_mask=global_mask
        )
        
        # --- Step 5: Output Projection ---
        node_pred = self.node_head(node_decoded)
        tri_pred = self.triangle_head(tri_decoded)
        
        return node_pred, tri_pred

    def predict_step(self, node_data, triangle_data):
        """
        封装用于推理的单步预测方法
        """
        self.eval()
        with torch.no_grad():
            return self.forward(node_data, triangle_data)


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
        EMBED_DIM = 128
        N_HEADS = 2
        NUM_LAYERS = 2      # 测试时减少层数以加快速度
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