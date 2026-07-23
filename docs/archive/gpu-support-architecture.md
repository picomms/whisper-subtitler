# GPU Support Architecture Design

🎨🎨🎨 **ENTERING CREATIVE PHASE: ARCHITECTURE DESIGN**

## Component Description

The GPU Support system is responsible for detecting available GPU acceleration platforms, selecting the optimal device for computation, and applying platform-specific optimizations. This component will enable whisper-subtitler to run efficiently on multiple GPU platforms including NVIDIA CUDA, AMD ROCm, and Apple Silicon Metal/MPS.

## Requirements & Constraints

### Functional Requirements

1. Support NVIDIA GPUs with CUDA
2. Support AMD GPUs with ROCm
3. Support Apple M-series chips with Metal Performance Shaders (MPS)
4. Provide fallback to CPU when GPU acceleration is unavailable
5. Automatically detect and select the optimal available GPU platform
6. Allow manual override of platform selection
7. Apply platform-specific optimizations for each backend

### Non-Functional Requirements

1. Maintain backward compatibility with existing use_cuda configuration
2. Minimize performance overhead for platform detection
3. Gracefully handle failures in GPU initialization
4. Provide clear error messages for GPU-related issues
5. Avoid unnecessary dependencies for unused platforms

### Constraints

1. Must work with existing PyTorch-based implementation
2. Must support PyAnnote's diarization pipeline on multiple platforms
3. Must be implemented without breaking existing functionality
4. Installation process should remain as simple as possible

## Multiple Options Analysis

### Option 1: Integrated Platform Support

**Description:**
Integrate platform support directly into the existing Transcriber and Diarizer classes, with conditional logic for different backends.

**Architecture:**
```
Config
├── use_cuda: bool  # Legacy setting
└── gpu_backend: str  # New setting: "auto", "cuda", "rocm", "mps", "cpu"

Transcriber/Diarizer
├── detect_platform()  # Internal method
├── select_device()    # Internal method
└── apply_optimizations()  # Internal method
```

**Pros:**
- Simple implementation with minimal new files
- Direct access to all configuration parameters
- No abstraction overhead

**Cons:**
- Mixes platform-specific code with business logic
- Classes become larger and harder to maintain
- Harder to test platform-specific code in isolation
- Duplication of platform detection logic across classes

### Option 2: Platform Abstraction Layer

**Description:**
Create a dedicated abstraction layer for platform operations with strategy-like patterns for different backends.

**Architecture:**
```
PlatformManager
├── detect_available_platforms()
├── select_optimal_platform()
└── get_platform(type)

Platform (Abstract)
├── initialize()
├── apply_optimizations()
└── cleanup()

CudaPlatform, RocmPlatform, MpsPlatform, CpuPlatform
└── [Platform-specific implementations]

Transcriber/Diarizer
└── Uses PlatformManager to get appropriate platform
```

**Pros:**
- Clean separation of platform-specific code
- Extensible for new platforms in the future
- Easier to test platform-specific code in isolation

**Cons:**
- More complex implementation
- Additional abstraction overhead
- Potential performance impact from abstraction
- Might be overkill for the current requirements

### Option 3: Platform Utilities Approach

**Description:**
Create utility functions for platform operations that can be used by any component when needed.

**Architecture:**
```
platform_utils.py
├── detect_available_backends()
├── get_optimal_device(config)
├── apply_platform_optimizations(device, model)
└── platform_specific_configs(platform)

Config
├── use_cuda: bool  # Legacy setting
└── gpu_backend: str  # New setting: "auto", "cuda", "rocm", "mps", "cpu"

Transcriber/Diarizer
└── Use platform_utils functions when needed
```

**Pros:**
- Simpler than full abstraction layer
- Keeps platform-specific code separate
- Easy to test
- Low overhead
- Practical implementation for current requirements

**Cons:**
- Less structured than a full abstraction layer
- Could lead to duplication if not carefully designed
- Less ideal for significant expansion of platform-specific code

## Recommended Approach

After analyzing the options, **Option 3: Platform Utilities Approach** provides the best balance of simplicity, maintainability, and performance for our requirements.

### Justification:

1. **Appropriate Complexity Level**: The utility-based approach provides enough structure without over-engineering the solution.
2. **Separation of Concerns**: Keeps platform-specific code separate from business logic.
3. **Testability**: Easy to test platform detection and selection in isolation.
4. **Low Overhead**: Minimal performance impact from the implementation.
5. **Practical Implementation**: Quickest path to supporting all required platforms.

The full abstraction layer of Option 2 would be more appropriate if we expected significant platform-specific code or planned to support many more platforms in the future. Option 1 doesn't provide enough separation of concerns and would make the codebase harder to maintain.

## Implementation Guidelines

### 1. Platform Utilities Module

Create a new `platform_utils.py` module with the following functions:

```python
def detect_available_backends():
    """
    Detect available GPU backends on the system.
    
    Returns:
        dict: Dictionary with backend names as keys and boolean availability as values
    """
    available = {"cpu": True}
    
    # Check for CUDA
    try:
        import torch
        available["cuda"] = torch.cuda.is_available()
    except ImportError:
        available["cuda"] = False
    
    # Check for ROCm
    try:
        import torch
        # Check if PyTorch was built with ROCm
        available["rocm"] = hasattr(torch, 'hip') and torch.hip.is_available()
    except (ImportError, AttributeError):
        available["rocm"] = False
    
    # Check for MPS (Apple Metal)
    try:
        import torch
        available["mps"] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        available["mps"] = False
    
    return available

def get_optimal_device(config):
    """
    Get the optimal device based on config and availability.
    
    Args:
        config: Configuration object with gpu_backend preference
        
    Returns:
        str: Backend name to use ("cuda", "rocm", "mps", or "cpu")
    """
    backends = detect_available_backends()
    
    # If auto, select the best available backend
    if config.gpu_backend == "auto":
        if backends["cuda"]:
            return "cuda"
        elif backends["rocm"]:
            return "rocm"
        elif backends["mps"]:
            return "mps"
        else:
            return "cpu"
    
    # If specific backend requested, check if available
    elif config.gpu_backend in backends:
        if backends[config.gpu_backend]:
            return config.gpu_backend
        else:
            logger.warning(f"Requested backend {config.gpu_backend} not available, falling back to CPU")
            return "cpu"
    
    # Default to CPU
    return "cpu"

def apply_platform_optimizations(backend, model=None):
    """
    Apply platform-specific optimizations.
    
    Args:
        backend: The GPU backend being used
        model: Optional model to optimize
    """
    import torch
    
    # CUDA optimizations
    if backend == "cuda":
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
    
    # ROCm optimizations
    elif backend == "rocm":
        torch.backends.cudnn.benchmark = True
        # ROCm-specific optimizations would go here
    
    # MPS optimizations
    elif backend == "mps":
        # Apple Metal optimizations would go here
        pass
```

### 2. Configuration Updates

Update the `Config` class to support the new GPU backend options:

```python
class Config:
    def __init__(self):
        # ... existing code ...
        
        # GPU settings
        self.use_cuda = True  # Legacy setting, keep for backward compatibility
        self.gpu_backend = "auto"  # auto, cuda, rocm, mps, cpu
        
        # ... existing code ...
    
    def load_from_env(self, env_file=None):
        # ... existing code ...
        
        # GPU settings
        self.gpu_backend = os.environ.get("GPU_BACKEND", self.gpu_backend)
        self.use_cuda = os.environ.get("USE_CUDA", str(self.use_cuda)).lower() in ("1", "true", "yes")
        
        # Backwards compatibility: if use_cuda is False, force CPU
        if not self.use_cuda and self.gpu_backend == "auto":
            self.gpu_backend = "cpu"
        
        # ... existing code ...
```

### 3. Integration in Transcriber/Diarizer

Update the Transcriber and Diarizer classes to use the platform utilities:

```python
# In Transcriber class
def __init__(self, config):
    # ... existing code ...
    
    # Determine device for inference using platform utilities
    from ..platform_utils import get_optimal_device
    self.backend = get_optimal_device(config)
    
    # Set device string based on backend
    if self.backend == "cuda" or self.backend == "rocm":
        self.device = "cuda"  # Both CUDA and ROCm use "cuda" device string in PyTorch
    elif self.backend == "mps":
        self.device = "mps"
    else:
        self.device = "cpu"
    
    # ... existing code ...

def load_model(self):
    # ... existing code ...
    
    # Apply platform-specific optimizations
    from ..platform_utils import apply_platform_optimizations
    apply_platform_optimizations(self.backend)
    
    # ... existing code ...
```

### 4. Dependency Management

Update `pyproject.toml` to define optional dependencies for different platforms:

```toml
[project]
name = "whisper-subtitler"
# ... existing content ...

[project.optional-dependencies]
cuda = [
    "torch>=2.0.0",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-cupti-cu12",
]
rocm = [
    "torch>=2.0.0+rocm",
]
mps = [
    "torch>=2.0.0",
]
```

### 5. Installation Instructions

Update the documentation to include installation instructions for different platforms:

```markdown
## Installation

### For NVIDIA GPU Support (Default)
```sh
pip install whisper-subtitler[cuda]
```

### For AMD GPU Support
```sh
pip install whisper-subtitler[rocm]
```

### For Apple Silicon Support
```sh
pip install whisper-subtitler[mps]
```

### For CPU-only
```sh
pip install whisper-subtitler
```
```

## Verification

This architecture satisfies all the requirements:

1. ✅ Supports all required GPU platforms (NVIDIA, AMD, Apple)
2. ✅ Provides automatic and manual platform selection
3. ✅ Maintains backward compatibility
4. ✅ Applies platform-specific optimizations
5. ✅ Gracefully handles platform unavailability
6. ✅ Keeps platform-specific code separate from business logic
7. ✅ Minimizes performance overhead
8. ✅ Makes dependencies platform-specific

🎨🎨🎨 **EXITING CREATIVE PHASE** 