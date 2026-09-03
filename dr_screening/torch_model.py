from __future__ import annotations


def build_model(model_name: str = "simple_cnn", num_classes: int = 5, pretrained: bool = False):
    """Build a PyTorch DR classifier.

    `simple_cnn` has no torchvision dependency. `resnet18` and `efficientnet_b0`
    use torchvision when it is installed.
    """

    try:
        import torch
        from torch import nn
    except (ImportError, OSError) as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for CNN training. Install torch and torchvision, "
            "or use scripts/train_sklearn.py for the lightweight baseline."
        ) from exc

    model_name = model_name.lower()

    if model_name in {"resnet18", "efficientnet_b0"}:
        try:
            from torchvision import models
        except (ImportError, OSError) as exc:  # pragma: no cover
            raise RuntimeError(f"{model_name} requires torchvision to be installed.") from exc

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model

        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model

    class SimpleFundusCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(256, 384, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(384),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.35),
                nn.Linear(384, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.20),
                nn.Linear(128, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return SimpleFundusCNN()
