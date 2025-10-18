"""
Data module initialization.

Role Assignments:
- Data Engineer: Primary owner of this module
- Software Engineer: Maintains module structure
"""

from src.data.data_loader import CSVDataLoader, DataFrameAnalyzer, VariableManager

__all__ = [
    'CSVDataLoader',
    'DataFrameAnalyzer',
    'VariableManager'
]
