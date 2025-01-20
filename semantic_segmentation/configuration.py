from dataclasses import dataclass


@dataclass
class SystemConfig:
    """
    From OpenCV University
    """

    # Seed number to set the state of all random number generators
    seed: int = 42

    # enable CuDNN benchmark for the sake of performance
    cudnn_benchmark_enabled: bool = False

    # Make cudnn deterministic (reproducible training)
    cudnn_deterministic: bool = True
