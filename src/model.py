"""
model.py
--------
WristNet: a from-scratch (no pretrained backbone, no torchvision.models)
residual CNN for multi-label classification of the 4 merged wrist-trauma
classes, designed specifically so that Class Activation Mapping (CAM) works
out of the box:

    conv stem -> [residual blocks with downsampling] -> Global Average Pool
              -> single Linear layer -> 4 sigmoid outputs

The CAM trick REQUIRES exactly this shape: the very last conv feature map
(before pooling) must connect to the output through nothing but GAP + one
Linear layer. That's why there's no extra hidden FC layer at the end --
adding one would break the direct spatial-to-class weight mapping CAM needs.

Every layer below is hand-defined with raw nn.Conv2d / nn.BatchNorm2d /
nn.ReLU -- this is "our own architecture", not an imported detector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Standard 2-conv residual block with a projection shortcut when the
    number of channels or the spatial stride changes."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class WristNet(nn.Module):
    """
    Input:  Bx3xHxW  (H=W=img_size, e.g. 256)
    Output: BxN_CLASSES raw logits (apply sigmoid outside for probabilities --
            kept as logits here so BCEWithLogitsLoss / focal loss can use them
            directly, which is more numerically stable than sigmoid+BCE).

    Also exposes `.last_feature_map` (set during forward) -- the BxCxhxw
    tensor right before global average pooling, which is exactly what CAM
    needs. And `.classifier.weight` (shape N_CLASSESxC) gives the per-class
    weights CAM combines with that feature map.
    """

    def __init__(self, n_classes: int = 4, in_channels: int = 3, base_width: int = 32,
                 dropout: float = 0.4, fine_cam: bool = True):
        super().__init__()

        # Stem: bring 256x256x3 down to 64x64 while extracting low-level edges
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # -> base_width x 64 x 64 (for 256 input)

        # 4 stages. With fine_cam=True the LAST stage does not downsample, so
        # the final feature map is 16x16 instead of 8x8 for a 256px input.
        # That matters because the CAM heatmap is computed at THIS resolution
        # and then upsampled: at 8x8 each CAM cell covers a 32x32 image block,
        # which is why the boxes came out coarse/blobby. 16x16 gives 4x more
        # spatial detail for localization, at the cost of extra compute in the
        # final stage only.
        stage4_stride = 1 if fine_cam else 2
        self.stage1 = self._make_stage(base_width, base_width, n_blocks=2, stride=1)          # 64x64
        self.stage2 = self._make_stage(base_width, base_width * 2, n_blocks=2, stride=2)      # 32x32
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, n_blocks=2, stride=2)  # 16x16
        self.stage4 = self._make_stage(base_width * 4, base_width * 8, n_blocks=2,
                                        stride=stage4_stride)  # 16x16 if fine_cam else 8x8

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)  # raised from 0.3 -- fight overfitting (train_loss kept
        # dropping while val AUROC stalled/wobbled, classic overfitting signature)
        self.classifier = nn.Linear(base_width * 8, n_classes)

        self.last_feature_map: torch.Tensor | None = None
        self._init_weights()

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_ch, out_ch, stride=stride)]
        for _ in range(n_blocks - 1):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        self.last_feature_map = x  # BxCxhxw, kept for CAM (Phase 2)

        pooled = self.gap(x).flatten(1)  # BxC
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # Bxn_classes
        return logits

    @torch.no_grad()
    def compute_cam(self, class_idx: int) -> torch.Tensor:
        """Compute the CAM heatmap for one class using the LAST forward
        pass's feature map. Call model(x) first, then this.

        Returns a Bxhxw tensor (not yet resized to the input image size or
        normalized to [0,1] -- that's done at visualization time in Phase 2).
        """
        if self.last_feature_map is None:
            raise RuntimeError("Call forward() before compute_cam().")
        weights = self.classifier.weight[class_idx]  # shape (C,)
        # weighted sum of feature-map channels -> BxhxW
        cam = torch.einsum("bchw,c->bhw", self.last_feature_map, weights)
        return cam


class FocalLoss(nn.Module):
    """
    Multi-label focal loss (sigmoid focal loss), built on top of
    BCEWithLogitsLoss. This is the mechanism that directly counters class
    imbalance: easy, well-classified examples (which dominate for the
    common classes like fracture) get down-weighted, and the loss focuses
    more on hard/rare examples (bone_lesion, foreign_material).

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma: focusing parameter (higher = more aggressive down-weighting of
           easy examples). 2.0 is the standard value from the original
           RetinaNet focal loss paper.
    alpha: per-class weight to further address imbalance. If None, computed
           automatically as inverse class frequency at construction time via
           `from_class_frequencies`.
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None, persistent=False)

    @classmethod
    def from_class_frequencies(cls, pos_counts: list[int], total: int, gamma: float = 2.0) -> "FocalLoss":
        """Build alpha per class as (1 - pos_rate), so rarer classes get a
        higher weight. pos_counts: number of positive images per class."""
        pos_rate = torch.tensor([c / total for c in pos_counts], dtype=torch.float32)
        alpha = 1.0 - pos_rate
        return cls(alpha=alpha, gamma=gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device) * targets + (1 - self.alpha.to(logits.device)) * (1 - targets)
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        return loss.mean()


if __name__ == "__main__":
    # Quick self-test: run `python src/model.py` -- checks shapes and that
    # CAM extraction works, using random data (no dataset needed for this test).
    print("[test] building WristNet...")
    model = WristNet(n_classes=4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] WristNet built. Total parameters: {n_params:,}")

    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print(f"[ok] forward pass: input {tuple(x.shape)} -> output {tuple(logits.shape)}")
    assert logits.shape == (2, 4), f"expected (2,4), got {logits.shape}"

    print(f"[ok] last_feature_map shape: {tuple(model.last_feature_map.shape)}")

    cam = model.compute_cam(class_idx=0)
    print(f"[ok] CAM for class 0 shape: {tuple(cam.shape)}")
    assert cam.shape[0] == 2 and cam.dim() == 3

    # focal loss sanity check
    targets = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])
    loss_fn = FocalLoss.from_class_frequencies(pos_counts=[13550, 713, 3154, 234], total=20327)
    loss = loss_fn(logits, targets)
    print(f"[ok] focal loss value: {loss.item():.4f}")
    print(f"     per-class alpha weights: {loss_fn.alpha.tolist()}")

    print("\n[done] model.py self-test passed.")
