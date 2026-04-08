"""
EdgeMAF-Net: Edge-aware Multi-scale Attention Fusion Network
"""
__version__ = "1.0.0"
__author__ = "zym"
__description__ = "EdgeMAF-Net: Edge-aware Multi-scale Attention Fusion Network for Image Classification"

from ultralytics import (
    ASSETS,
    checks,
    download,
    settings,
    Explorer,
)
from ultralytics import YOLO as EdgeMAF
from ultralytics import (
    RTDETR,
    SAM,
    YOLOWorld,
    FastSAM,
    NAS,
)

__all__ = [
    "__version__",
    "__author__",
    "__description__",
    "EdgeMAF",
    "RTDETR",
    "SAM",
    "YOLOWorld",
    "FastSAM",
    "NAS",
    "ASSETS",
    "checks",
    "download",
    "settings",
    "Explorer",
]

def info():
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║              EdgeMAF-Net v{__version__}                      ║
    ║  Edge-aware Multi-scale Attention Fusion Network        ║
    ║                            ║
    ╚══════════════════════════════════════════════════════════╝
    """)

if __name__ != "__main__":
    pass  
