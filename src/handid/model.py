"""The model: pretrained backbone -> embedding head -> ArcFace classification loss.

At test time the classifier is thrown away and hands are compared by the cosine
similarity of their embeddings, so the model works for people it has never seen.
"""

from __future__ import annotations

import math

import lightning as L
import timm
import torch
import torch.nn.functional as F
from torch import nn


class ArcFace(nn.Module):
    """Additive angular margin loss (Deng et al., 2019).

    logits = s * cos(theta + m) for the true class, s * cos(theta) otherwise, where theta
    is the angle between the embedding and the class centre. The margin m forces
    embeddings of one identity to cluster tightly, which is exactly what verification needs.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float = 30.0, margin: float = 0.3):
        super().__init__()
        self.centres = nn.Parameter(torch.randn(num_classes, embedding_dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = F.linear(F.normalize(emb), F.normalize(self.centres)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = F.one_hot(labels, cos.shape[1]).bool()
        logits = self.scale * torch.where(target, torch.cos(theta + self.margin), cos)
        return F.cross_entropy(logits, labels), cos


class PartFusion(nn.Module):
    """Embedding from LOCAL features of a convolutional feature map.

    The aligned hand always has the same layout (fingers at the top, knuckles in the middle,
    back of the hand at the bottom), so a fixed grid of regions roughly corresponds to hand
    parts. Each region becomes a token (average of its feature-map cells + a learned position
    embedding); tokens are fused by

        mean          a small MLP per token, then the average  (order-free "deep sets" fusion)
        transformer   2 transformer layers with a CLS token     (regions can attend to each other)

    Compared with global average pooling of the whole map, the network keeps track of WHERE a
    feature was seen, which matters for biometrics (a crease on the index knuckle is not the
    same evidence as the same crease on the little finger).
    """

    def __init__(self, in_dim: int, dim: int, grid: int = 4, fusion: str = "transformer", layers: int = 2):
        super().__init__()
        self.grid, self.fusion = grid, fusion
        self.proj = nn.Linear(in_dim, dim)
        self.pos = nn.Parameter(torch.randn(1, grid * grid, dim) * 0.02)
        if fusion == "mean":
            self.token_mlp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        elif fusion == "transformer":
            self.cls = nn.Parameter(torch.zeros(1, 1, dim))
            layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=4 * dim, dropout=0.1,
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"unknown fusion {fusion}")
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:  # B x C x h x w
        parts = F.adaptive_avg_pool2d(fmap, self.grid).flatten(2).transpose(1, 2)  # B x grid^2 x C
        tokens = self.proj(parts) + self.pos
        if self.fusion == "mean":
            fused = (tokens + self.token_mlp(tokens)).mean(dim=1)
        else:
            x = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)
            fused = self.norm(self.encoder(x)[:, 0])
        return self.bn(fused)


class HandEmbedder(nn.Module):
    """timm backbone + embedding head.

    Input: float images in [0, 1]. Normalisation uses the statistics the backbone was
    pre-trained with (ImageNet for DINO, 0.5/0.5 for SigLIP, CLIP's own for CLIP).

    pooling
        global        the backbone's own pooled features -> Linear -> BatchNorm  (default)
        parts_mean    local regions of the CNN feature map, averaged (see PartFusion)
        parts_transformer   local regions fused by a small transformer
    embedding_dim=None -> raw backbone features (zero-shot baseline).
    """

    def __init__(self, backbone: str, embedding_dim: int | None, pretrained: bool = True,
                 timm_kwargs: dict | None = None, pooling: str = "global", grid: int = 4):
        super().__init__()
        self.pooling = pooling
        extra = {} if pooling == "global" else {"global_pool": ""}  # keep the spatial feature map
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                          **extra, **(timm_kwargs or {}))
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        feat = self.backbone.num_features
        if pooling != "global":
            self.head = PartFusion(feat, embedding_dim, grid, fusion=pooling.removeprefix("parts_"))
        elif embedding_dim:
            self.head = nn.Sequential(nn.Linear(feat, embedding_dim), nn.BatchNorm1d(embedding_dim))
        else:
            self.head = nn.Identity()
        self.dim = embedding_dim or feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        if self.pooling == "global":
            return self.head(self.backbone(x))
        return self.head(self.backbone.forward_features(x))


class HandIDModule(L.LightningModule):
    def __init__(self, num_classes: int, backbone: str, embedding_dim: int | None, arcface_scale: float,
                 arcface_margin: float, lr_backbone: float, lr_head: float, weight_decay: float,
                 warmup_epochs: int, layer_decay: float = 1.0, timm_kwargs: dict | None = None,
                 metadata_classes: dict[str, int] | None = None, metadata_weight: float = 0.1,
                 pooling: str = "global", grid: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.net = HandEmbedder(backbone, embedding_dim, timm_kwargs=timm_kwargs, pooling=pooling, grid=grid)
        self.loss = ArcFace(self.net.dim, max(num_classes, 1), arcface_scale, arcface_margin)
        # Optional auxiliary heads: predict metadata (gender, skin colour, ...) from the embedding.
        self.metadata_heads = nn.ModuleDict({attr: nn.Linear(self.net.dim, n)
                                             for attr, n in (metadata_classes or {}).items()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x).float(), dim=1)

    def training_step(self, batch, _):
        images, labels, metadata = batch
        emb = self.net(images)
        loss, cos = self.loss(emb, labels)
        logs = {"train/arcface": loss, "train/acc": (cos.argmax(1) == labels).float().mean()}
        for k, (attr, head) in enumerate(self.metadata_heads.items()):
            known = metadata[:, k] >= 0
            if known.any():  # only 11k rows carry metadata
                aux = F.cross_entropy(head(emb[known]).float(), metadata[known, k])
                loss = loss + self.hparams.metadata_weight * aux
                logs[f"train/meta_{attr}"] = aux
        logs["train/loss"] = loss
        self.log_dict(logs, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        hp = self.hparams
        param_groups = self._backbone_param_groups(hp.lr_backbone, hp.layer_decay, hp.weight_decay)
        param_groups.append({"params": [*self.net.head.parameters(), *self.loss.parameters(),
                                        *self.metadata_heads.parameters()],
                             "lr": hp.lr_head, "weight_decay": hp.weight_decay})
        opt = torch.optim.AdamW(param_groups)
        total = self.trainer.estimated_stepping_batches
        warmup = int(total * hp.warmup_epochs / max(self.trainer.max_epochs, 1))

        def lr_factor(step: int) -> float:  # linear warm-up, then cosine decay to 0
            if step < warmup:
                return (step + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total - warmup, 1)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def _backbone_param_groups(self, lr: float, layer_decay: float, wd: float) -> list[dict]:
        """Lower learning rates for earlier ViT blocks (layer-wise decay); 1.0 disables it."""
        blocks = getattr(self.net.backbone, "blocks", None)
        n = len(blocks) if blocks is not None else 0
        groups: dict[int, list] = {}
        for name, p in self.net.backbone.named_parameters():
            if name.startswith("blocks."):
                depth = int(name.split(".")[1]) + 1
            elif name.startswith(("norm", "fc_norm", "head")):
                depth = n + 1
            else:  # patch embedding, cls token, position embedding
                depth = 0
            groups.setdefault(depth, []).append(p)
        return [{"params": ps, "lr": lr * layer_decay ** (n + 1 - d), "weight_decay": wd}
                for d, ps in sorted(groups.items())]
