"""
cross_modal_knowledge_enhancer.py
==================================
跨模态知识增强模块 —— Reg2RG 报告生成系统的核心 Cross-Attention 增强组件。

当前架构（无在线检索版）：
    报告向量由 MyEmbedding._select_organ_annotations + _lookup_annotation_vectors
    在 forward 入口处从离线 NPZ Bank（feats [N_patients, N_organs, bank_dim]）中
    直接查表取出，每个器官预选 5 个最近邻报告向量，作为 organ_report_vectors 传入。
    CrossModalKnowledgeEnhancer 和 RWLKE 仅负责 Cross-Attention 增强，
    不再做任何在线相似度检索。

数据流：
    organ_report_vectors: list[dict[organ, Tensor[num_reports, bank_dim]]]
        │
        ├─► [KE / CrossModalKnowledgeEnhancer.forward]
        │       拼接所有器官报告向量 → 对整图视觉 Token 做 Cross-Attention
        │
        └─► [RWLKE / RegionWiseLocalKnowledgeEnhancer.forward]
                按器官切片，对各器官的 Patch Token 段独立做 Cross-Attention

包含以下组件：
    1. CrossModalKnowledgeEnhancer (KE)         —— 全局跨模态 Cross-Attention 增强
    2. RegionWiseLocalKnowledgeEnhancer (RWLKE) —— 区域级多器官增强管理器
    3. KANEnhancedSpatialPooling (K-SAP)        —— KAN 空间自适应注意力池化（待接入）
    4. GlobalKnowledgeEnhancerWithKSAP (GKE)    —— K-SAP + GCN 全局增强（待接入）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# =====================================================================
# 模块 1：单器官 / 全局跨模态知识增强 (Cross-Modal Knowledge Enhancer / KE)
# =====================================================================

class CrossModalKnowledgeEnhancer(nn.Module):
    """
    跨模态知识增强器 —— 让视觉特征 Token 通过 Cross-Attention "阅读"报告文本特征，
    将历史病例报告中的领域先验知识注入到视觉表示中。

    报告向量（organ_report_vectors）由上游 MyEmbedding 从离线 NPZ Bank 中查表获得，
    本模块不做任何在线检索，直接使用传入的向量。

    两种调用方式：
        1. forward(visual_tokens, organ_report_vectors)
               —— 全局增强入口：将所有器官的报告向量拼接后对整图视觉 Token 做增强。
        2. _cross_attend(organ_tokens, selected_vectors)
               —— 局部增强接口：供 RWLKE 对单器官 Patch Token 段直接调用。

    Cross-Attention 计算：
        Q = linear(visual_tokens)           ← 视觉特征作为 Query
        K = linear(report_vectors)          ← 报告特征作为 Key
        V = linear(report_vectors)          ← 报告特征作为 Value
        out = softmax(Q·Kᵀ / √d_head) · V
        output = LayerNorm(Q_residual + out) → fc1

    Args:
        d_model  (int): 视觉 Token 的特征维度（Cross-Attention 的工作维度）
        bank_dim (int): 报告向量的维度（NPZ feats 的最后一维，当前为 256）
        num_heads (int): 多头注意力头数，默认 8
        dropout  (float): 注意力权重 Dropout 概率，默认 0.1
    """

    def __init__(self, d_model, bank_dim, num_heads=8, dropout=0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.bank_dim = bank_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # 缩放因子 1/√d_head，防止点积分数过大导致 softmax 梯度消失
        self.scale = (d_model // num_heads) ** -0.5

        # ── Cross-Attention 三个核心投影层 ──
        self.visual_query      = nn.Linear(d_model, d_model)   # Q：视觉特征 → Query
        self.top_k_report_key  = nn.Linear(bank_dim, d_model)  # K：报告特征 → Key
        self.top_k_report_value = nn.Linear(bank_dim, d_model) # V：报告特征 → Value

        # ── 后处理：残差归一化 + 非线性增强 ──
        self.norm = nn.LayerNorm(d_model)
        self.fc1  = nn.Linear(d_model, d_model)

    def forward(self, visual_tokens, organ_report_vectors):
        """
        全局增强入口：将当前样本所有器官的预选报告向量拼接后，
        对整图视觉 Token 做 Cross-Attention 知识增强。

        Args:
            visual_tokens (Tensor): 整图视觉特征，形状 [B, visual_len, d_model]
            organ_report_vectors (list[dict]): 长度为 B 的列表，每个元素是
                {organ_name: Tensor[num_reports, bank_dim]}，
                由 MyEmbedding._lookup_annotation_vectors 从 NPZ Bank 查表得到。

        Returns:
            Tensor: Cross-Attention 增强后的视觉特征，形状 [B, visual_len, d_model]
        """
        if organ_report_vectors is None or len(organ_report_vectors) != visual_tokens.size(0):
            raise ValueError("organ_report_vectors must contain one mapping per batch sample")

        enhanced = []
        for index, sample_organ_vectors in enumerate(organ_report_vectors):
            sample_tokens = visual_tokens[index:index + 1]  # [1, visual_len, d_model]

            # 过滤无报告的器官，将剩余器官的报告向量拼接为统一的 key/value 集合
            available = [v for v in sample_organ_vectors.values() if v.numel() > 0]
            if not available:
                # 当前样本无任何可用报告向量，保留原始视觉特征
                enhanced.append(sample_tokens)
                continue

            # [total_reports, bank_dim] → unsqueeze → [1, total_reports, bank_dim]
            selected = torch.cat(available, dim=0)
            selected = selected.to(device=sample_tokens.device, dtype=sample_tokens.dtype).unsqueeze(0)

            enhanced.append(self._cross_attend(sample_tokens, selected))

        return torch.cat(enhanced, dim=0)  # [B, visual_len, d_model]

    def _cross_attend(self, visual_tokens, report_tokens):
        """
        多头 Cross-Attention 核心计算。
        视觉特征作为 Query，报告文本特征作为 Key/Value。

        计算步骤：
            Q [B, H, visual_len, d_head]  ← visual_tokens
            K [B, H, d_head, num_reports] ← report_tokens（已转置）
            V [B, H, num_reports, d_head] ← report_tokens
            Attn = softmax(Q·K / √d_head) · V
            out  = LayerNorm(Q_residual + Attn) → fc1

        Args:
            visual_tokens (Tensor): 视觉特征，形状 [B, visual_len, d_model]
            report_tokens (Tensor): 预选报告特征，形状 [B, num_reports, bank_dim]
                                    num_reports 通常为 5（每器官最近邻数）×实际器官数

        Returns:
            Tensor: 增强后特征，形状 [B, visual_len, d_model]
        """
        batch_size = visual_tokens.size(0)
        d_head = self.d_model // self.num_heads

        # ── Step 1：Q 投影 + 多头分割 → [B, num_heads, visual_len, d_head] ──
        query = self.visual_query(visual_tokens)               # [B, visual_len, d_model]
        res_query = query.view(batch_size, -1, self.d_model)   # 保存残差副本
        query = query.view(batch_size, -1, self.num_heads, d_head).permute(0, 2, 1, 3)

        # ── Step 2：K 投影 + 多头分割（转置备用）→ [B, num_heads, d_head, num_reports] ──
        key = self.top_k_report_key(report_tokens)             # [B, num_reports, d_model]
        key = key.view(batch_size, -1, self.num_heads, d_head).permute(0, 2, 3, 1)

        # ── Step 3：V 投影 + 多头分割 → [B, num_heads, num_reports, d_head] ──
        value = self.top_k_report_value(report_tokens)         # [B, num_reports, d_model]
        value = value.view(batch_size, -1, self.num_heads, d_head).permute(0, 2, 1, 3)

        # ── Step 4：缩放点积注意力 ──
        # attn[b, h, i, j] = softmax( Q[b,h,i,:] · K[b,h,:,j] / √d_head )
        attn = torch.matmul(query, key) * self.scale           # [B, num_heads, visual_len, num_reports]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ── Step 5：加权聚合 + 残差归一化 + 非线性增强 ──
        out = torch.matmul(attn, value)                        # [B, num_heads, visual_len, d_head]
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        out = self.norm(res_query + out)                       # 残差 + LayerNorm
        out = self.fc1(F.gelu(out))                           # GELU 非线性 + 线性增强
        return out  # [B, visual_len, d_model]


# =====================================================================
# 模块 2：区域级局部知识增强管理器 (RWLKE - Region-Wise Local Knowledge Enhancer)
# =====================================================================

class RegionWiseLocalKnowledgeEnhancer(nn.Module):
    """
    区域级局部知识增强器 —— 为每个器官独立维护一个 CrossModalKnowledgeEnhancer，
    对各器官的 Patch Token 段分别用该器官专属报告向量做 Cross-Attention 增强，
    避免不同器官知识相互干扰。

    输入布局（vision_region_embedding）：
        所有器官的 Patch Token 在序列维度上顺序拼接：
        [organ_0_tokens | organ_1_tokens | ... | organ_N_tokens]
        每个器官占 region_token_len（默认 33）个 Token 位置。

    增强结果原位写回，返回与输入等形状的张量。

    Args:
        organs_list      (list[str]): 所有器官名称，与 NPZ bank 的 organs 字段对齐
        d_model          (int): 视觉 Token 特征维度
        bank_dim         (int): 报告向量维度（NPZ feats 最后一维，当前为 256）
        num_heads        (int): 多头注意力头数，默认 8
        dropout          (float): Dropout 概率，默认 0.1
        region_token_len (int): 每器官在序列中占用的 Token 数，默认 33
    """

    def __init__(self, organs_list, d_model, bank_dim,
                 num_heads=8, dropout=0.1, region_token_len=33):
        super().__init__()
        self.organs_list = organs_list
        self.region_token_len = region_token_len

        # 每个器官独立一个增强器，nn.ModuleDict 确保参数被正确注册
        self.enhancers = nn.ModuleDict({
            organ: CrossModalKnowledgeEnhancer(d_model, bank_dim, num_heads, dropout)
            for organ in organs_list
        })

    def forward(self, vision_region_embedding, region2areas, organ_report_vectors):
        """
        遍历每个样本的每个器官，用该器官预选报告向量对对应 Token 段做 Cross-Attention。

        Args:
            vision_region_embedding (Tensor): 拼接的区域视觉特征，
                                              形状 [B, num_organs * region_token_len, d_model]
            region2areas (list[dict|list]):   长度为 B，记录各样本的器官名称与位置顺序
            organ_report_vectors (list[dict]): 长度为 B，
                                              {organ: Tensor[num_reports, bank_dim]}，
                                              由 MyEmbedding._lookup_annotation_vectors 提供

        Returns:
            Tensor: 逐器官增强后的特征，形状同输入 [B, num_organs * region_token_len, d_model]
        """
        if organ_report_vectors is None or len(organ_report_vectors) != vision_region_embedding.size(0):
            raise ValueError("organ_report_vectors must contain one mapping per batch sample")

        # clone 避免原地修改影响计算图
        enhanced = vision_region_embedding.clone()
        #---Dan---
        touched_organs = set()
        #---Dan---

        for i, regions_for_sample in enumerate(region2areas):
            organ_names = regions_for_sample.values() if isinstance(regions_for_sample, dict) \
                          else regions_for_sample

            for j, organ in enumerate(organ_names):
                # 从预选向量中取当前器官的报告向量
                vectors = organ_report_vectors[i].get(organ)
                if vectors is None or vectors.numel() == 0:
                    # 该器官在 Bank 中无匹配记录，跳过
                    continue

                # 定位该器官在拼接序列中的 Token 区间
                start = j * self.region_token_len
                end   = (j + 1) * self.region_token_len
                organ_tokens = vision_region_embedding[i:i + 1, start:end, :]
                # [1, region_token_len, d_model]

                # 对齐设备/dtype，升维对齐 batch 维度 → [1, num_reports, bank_dim]
                selected = vectors.to(device=organ_tokens.device,
                                      dtype=organ_tokens.dtype).unsqueeze(0)

                # 用该器官专属增强器做 Cross-Attention，原位写回
                enhanced[i:i + 1, start:end, :] = self.enhancers[organ]._cross_attend(
                    organ_tokens, selected
                )
                touched_organs.add(organ)

        # ---Dan---
        # 多卡训练时，DeepSpeed 要求每张卡这一次反向传播产生的梯度张量集合大小完全一致，
        # 否则 all-reduce 会因为各卡实际传的数据量不一样而永久卡死。self.enhancers 是按
        # 器官分开的独立子模块，如果这个 batch 里某个器官在任何样本里都没出现（在数据量小、
        # 器官种类多时很常见），它的参数这一步就完全不会进入计算图，导致这张卡的梯度比
        # 别的卡少一块。这里让没被用到的器官也过一遍全零输入的 dummy 前向，乘 0 后加回
        # enhanced（不改变任何实际数值），让它的参数照样产生一个真实的（全零）梯度，
        # 保证每张卡的梯度张量大小恒定。
        missing_organs = [organ for organ in self.organs_list if organ not in touched_organs]
        for organ in missing_organs:
            dummy_tokens = enhanced.new_zeros((1, self.region_token_len, enhanced.size(-1)))
            dummy_reports = enhanced.new_zeros((1, 1, self.enhancers[organ].bank_dim))
            dummy_out = self.enhancers[organ]._cross_attend(dummy_tokens, dummy_reports)
            enhanced = enhanced + dummy_out.sum() * 0.0
        # ---Dan---

        return enhanced  # [B, num_organs * region_token_len, d_model]


# =====================================================================
# 模块 3：空间均值池化（KAN 暂时禁用）
# =====================================================================

# KAN 版本按实验设计保留，但当前运行路径不启用：
# class KANEnhancedSpatialPooling(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.kan_gate = KANLinear(d_model, 1)
#     def forward(self, enhanced_local_features_dict):
#         ...  # KAN patch scoring + weighted pooling

class SpatialMeanPooling(nn.Module):
    """将每个器官的 patch token 均值池化为一个 GCN 节点。"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enhanced_local_features_dict):
        if not enhanced_local_features_dict:
            raise ValueError("enhanced_local_features_dict must not be empty")

        organ_names = list(enhanced_local_features_dict.keys())
        pooled_organ_list = []
        batch_size = None
        for organ in organ_names:
            feat = enhanced_local_features_dict[organ]
            if feat.ndim != 3 or feat.size(1) == 0 or feat.size(-1) != self.d_model:
                raise ValueError(f"Invalid features for {organ!r}: {tuple(feat.shape)}")
            if batch_size is None:
                batch_size = feat.size(0)
            elif feat.size(0) != batch_size:
                raise ValueError("All organ features must have the same batch size")
            pooled_feat = feat.mean(dim=1)
            pooled_organ_list.append(self.norm(self.proj(pooled_feat)))

        return torch.stack(pooled_organ_list, dim=1), organ_names


# Backward-compatible name for older checkpoints/configuration code.
KANEnhancedSpatialPooling = SpatialMeanPooling


# =====================================================================
# 模块 4：全局知识增强与解剖拓扑图网络 (GKE) —— 待接入
# =====================================================================

class GlobalKnowledgeEnhancerWithKSAP(nn.Module):
    """
    全局知识增强器（GKE）—— 三阶段流程：
        1. K-SAP 池化：将各器官 Patch Token 压缩为单一节点向量
        2. Lift 加权聚合（Pre-GCN）：根据器官异常共现 Lift 概率矩阵，
           让相关性高的器官先互相聚合特征，注入领域先验拓扑知识
        3. GCN：在 Lift 增强后的节点上学习器官间的结构化拓扑关联

    Lift 矩阵（lift_matrix_path）：
        由 CT 报告中器官异常条件概率计算的 lift 值矩阵（N_organs × N_organs）。
        lift[i,j] = P(organ_i 异常 ∩ organ_j 异常) / (P(organ_i 异常) × P(organ_j 异常))
        值 > 1 表示正相关，= 1 表示独立，< 1 表示负相关。
        对角线（自身 lift）在此处置零，仅关注跨器官关联。

    注意：此模块依赖 KANLinear 和 GCNConv（PyG），需在引入前安装对应依赖。

    Args:
        organs_list (list[str]): 器官名称列表（与 REGIONS 对齐，决定 lift_adj 行列对应关系）
        d_model     (int): 特征维度
    """
    ORGAN_ORDER = (
        "abdomen", "bone", "breast", "esophagus", "heart", "lung",
        "mediastinum", "pleura", "thyroid", "trachea and bronchie",
    )

    # Lift 矩阵原始值，来自 CT-RATE 数据集 organ_lift_qwen_v4_21_of_26 统计结果。
    # 行列顺序与 REGIONS 一致：
    #   abdomen, bone, breast, esophagus, heart, lung,
    #   mediastinum, pleura, thyroid, trachea and bronchie
    _LIFT_RAW = [
        # abd        bone       breast     esoph      heart      lung       medias     pleura     thyroid    trachea
        [2.400591,  1.579119,  1.287274,  1.392178,  1.172228,  1.083063,  1.542688,  1.502302,  1.680172,  1.357497],  # abdomen
        [1.579119,  3.047857,  1.300615,  1.551256,  1.357132,  1.094797,  1.688777,  1.561668,  1.786357,  1.459479],  # bone
        [1.287274,  1.300615,  31.381643, 1.257181,  0.949651,  1.025445,  1.225845,  1.686433,  1.169306,  1.000745],  # breast
        [1.392178,  1.551256,  1.257181,  7.435330,  1.287193,  1.107171,  1.417956,  1.327606,  1.871936,  1.455621],  # esophagus
        [1.172228,  1.357132,  0.949651,  1.287193,  3.829436,  1.059950,  1.617845,  1.533481,  1.442305,  1.394897],  # heart
        [1.083063,  1.094797,  1.025445,  1.107171,  1.059950,  1.170591,  1.123692,  1.140732,  1.065674,  1.150828],  # lung
        [1.542688,  1.688777,  1.225845,  1.417956,  1.617845,  1.123692,  3.903846,  1.850411,  1.867399,  1.686296],  # mediastinum
        [1.502302,  1.561668,  1.686433,  1.327606,  1.533481,  1.140732,  1.850411,  4.826152,  1.667039,  1.492124],  # pleura
        [1.680172,  1.786357,  1.169306,  1.871936,  1.442305,  1.065674,  1.867399,  1.667039,  19.625378, 1.342049],  # thyroid
        [1.357497,  1.459479,  1.000745,  1.455621,  1.394897,  1.150828,  1.686296,  1.492124,  1.342049,  3.323329],  # trachea and bronchie
    ]

    def __init__(self, organs_list, d_model):
        super().__init__()
        if tuple(organs_list) != self.ORGAN_ORDER:
            raise ValueError(
                "organs_list must match the canonical organ order: "
                f"{self.ORGAN_ORDER}, got {tuple(organs_list)}"
            )
        self.organs_list = list(organs_list)
        self.pooler = SpatialMeanPooling(d_model)

        lift_matrix = torch.tensor(self._LIFT_RAW, dtype=torch.float32)
        lift_matrix.fill_diagonal_(0.0)
        lift_matrix = (lift_matrix - 1.0).clamp(min=0.0)
        lift_adj = lift_matrix / lift_matrix.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        self.register_buffer("lift_adj", lift_adj)
        self.lift_proj = nn.Linear(d_model, d_model)
        self.lift_norm = nn.LayerNorm(d_model)
        self.lift_gate = nn.Linear(d_model, 1)

        self.gcn1 = GCNConv(d_model, d_model)
        self.gcn2 = GCNConv(d_model, d_model)
        self.global_proj = nn.Linear(d_model, d_model)



    @staticmethod
    def _edge_index_from_lift(lift_adj):
        # lift_adj[target, source] maps to PyG edge_index[source, target].
        target, source = torch.nonzero(lift_adj > 0, as_tuple=True)
        return torch.stack((source, target), dim=0)

    def forward(self, enhanced_local_features_dict, edge_index=None):
        """Mean-pool organ tokens, apply gated LIFT aggregation, then two GCN layers."""
        active_organs = [
            organ for organ in self.organs_list if organ in enhanced_local_features_dict
        ]
        if not active_organs:
            raise ValueError("No known organs were provided to the global enhancer")

        ordered_dict = {organ: enhanced_local_features_dict[organ] for organ in active_organs}
        node_inputs, pooled_names = self.pooler(ordered_dict)
        if pooled_names != active_organs:
            raise RuntimeError("Pooling changed the canonical organ order")

        active_indices = torch.tensor(
            [self.organs_list.index(organ) for organ in active_organs],
            device=node_inputs.device,
            dtype=torch.long,
        )
        active_lift_adj = self.lift_adj.index_select(0, active_indices).index_select(
            1, active_indices
        )
        active_lift_adj = active_lift_adj / active_lift_adj.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)
        active_lift_adj = active_lift_adj.to(dtype=node_inputs.dtype)

        lift_context = torch.matmul(active_lift_adj.unsqueeze(0), node_inputs)
        gate = torch.sigmoid(self.lift_gate(node_inputs))
        node_inputs = self.lift_norm(
            node_inputs + gate * self.lift_proj(lift_context)
        )


        if edge_index is None:
            edge_index = self._edge_index_from_lift(active_lift_adj)
        edge_index = edge_index.to(device=node_inputs.device, dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
        if edge_index.numel() and int(edge_index.max()) >= len(active_organs):
            raise ValueError("edge_index contains a node outside the active organ set")

        gcn_outputs = []
        for sample_nodes in node_inputs:
            nodes = F.relu(self.gcn1(sample_nodes, edge_index))
            nodes = self.gcn2(nodes, edge_index)
            gcn_outputs.append(nodes)
        updated_nodes = torch.stack(gcn_outputs, dim=0)
        global_anatomical_features = self.global_proj(updated_nodes.mean(dim=1))
        return global_anatomical_features, updated_nodes
