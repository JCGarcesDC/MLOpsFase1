"""
Base module for core abstract classes and interfaces.

This module provides foundational abstract base classes following SOLID principles
for data science and ML engineering workflows.

Role Assignments:
- Software Engineer: Maintains base class architecture and design patterns
- ML Engineer: Extends base classes for specific ML workflows
- Data Engineer: Implements data pipeline interfaces
- Data Scientist: Uses concrete implementations for experiments
"""

from src.base.base_data_loader import BaseDataLoader
from src.base.base_preprocessor import BasePreprocessor
from src.base.base_model import BaseModel
from src.base.base_pipeline import BasePipeline

__all__ = [
    'BaseDataLoader',
    'BasePreprocessor',
    'BaseModel',
    'BasePipeline'
]
