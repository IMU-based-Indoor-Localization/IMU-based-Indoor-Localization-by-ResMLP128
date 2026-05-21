import torch


class DiagonalParam:
    @staticmethod
    def vec2Cov(vec):
        """log-variance 벡터 [B, 3] → 대각 공분산 행렬 [B, 3, 3]"""
        return torch.diag_embed(torch.exp(vec))
