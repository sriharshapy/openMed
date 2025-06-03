"""
OpenMed Utils Package
Contains utility functions and classes for OpenMed.
"""

from .gradcam import (
    GradCAM,
    ModelGradCAM,
    overlay_heatmap,
    visualize_gradcam,
    batch_gradcam_analysis,
    create_gradcam_for_model
)

__all__ = [
    # GradCAM functionality
    'GradCAM',
    'ModelGradCAM',
    'overlay_heatmap',
    'visualize_gradcam',
    'batch_gradcam_analysis',
    'create_gradcam_for_model'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "OpenMed Team" 