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


class HandEmbedder(nn.Module):
    """timm backbone + (optional) linear projection to an embedding.

    Input: float images in [0, 1]. Normalisation uses the statistics the backbone was
    pre-trained with (ImageNet for DINO, 0.5/0.5 for SigLIP, CLIP's own for CLIP).
    """

    def __init__(self, backbone: str, embedding_dim: int | None, pretrained: bool = True,
                 timm_kwargs: dict | None = None):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, **(timm_kwargs or {}))
        cfg = self.backbone.pretrained_cfg
        self.register_buffer("mean", torch.tensor(cfg["mean"]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(cfg["std"]).view(1, 3, 1, 1), persistent=False)
        feat = self.backbone.num_features
        # embedding_dim=None -> use raw backbone features (zero-shot baseline)
        self.head = nn.Sequential(nn.Linear(feat, embedding_dim), nn.BatchNorm1d(embedding_dim)) \
            if embedding_dim else nn.Identity()
        self.dim = embedding_dim or feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone((x - self.mean) / self.std))


class HandIDModule(L.LightningModule):
    def __init__(self, num_classes: int, backbone: str, embedding_dim: int | None, arcface_scale: float,
                 arcface_margin: float, lr_backbone: float, lr_head: float, weight_decay: float,
                 warmup_epochs: int, layer_decay: float = 1.0, timm_kwargs: dict | None = None):
        super().__init__()
        self.save_hyperparameters()
        self.net = HandEmbedder(backbone, embedding_dim, timm_kwargs=timm_kwargs)
        self.loss = ArcFace(self.net.dim, max(num_classes, 1), arcface_scale, arcface_margin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x).float(), dim=1)

    def training_step(self, batch, _):
        images, labels = batch
        loss, cos = self.loss(self.net(images), labels)
        acc = (cos.argmax(1) == labels).float().mean()
        self.log_dict({"train/loss": loss, "train/acc": acc}, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        hp = self.hparams
        param_groups = self._backbone_param_groups(hp.lr_backbone, hp.layer_decay, hp.weight_decay)
        param_groups.append({"params": [*self.net.head.parameters(), *self.loss.parameters()],
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
