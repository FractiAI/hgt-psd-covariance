"""Hierarchical Genomic Tokenization + Structured PSD Covariance Operators."""

__all__ = ["StructuredPSDModel", "StratifiedFrobeniusLoss", "HierarchicalGenomicTokenizer"]


def __getattr__(name: str):
    if name == "StructuredPSDModel":
        from hgt_psd.model import StructuredPSDModel
        return StructuredPSDModel
    if name == "StratifiedFrobeniusLoss":
        from hgt_psd.loss import StratifiedFrobeniusLoss
        return StratifiedFrobeniusLoss
    if name == "HierarchicalGenomicTokenizer":
        from hgt_psd.tokenization import HierarchicalGenomicTokenizer
        return HierarchicalGenomicTokenizer
    raise AttributeError(name)
