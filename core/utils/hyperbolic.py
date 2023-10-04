import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from geoopt.manifolds import PoincareBall
from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter

PROJ_EPS = 1e-3


class HyperMapper(object):
    """A class to map between euclidean and hyperbolic space and compute distances."""

    def __init__(self, c=1.0) -> None:
        """Initialize the hyperbolic mapper.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0
        """
        self.pb = PoincareBall(c=c).cuda()

    def expmap(self, x: torch.Tensor, dim=-1):
        """Exponential mapping from Euclidean to hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        x_hyp = self.pb.expmap0(x.double(), dim=dim).cuda(x.device)
        return x_hyp

    def logmap(self, x, dim=-1):
        """Logarithmic mapping from hyperbolic to Euclidean space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            dim (int, optional): Dimension to compute the logmap. Defaults to -1.

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        return self.pb.logmap0(x.double(), dim=dim).cuda(x.device)

    def poincare_distance(self, x, y, dim=-1):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)
            dim (int, optional): Dimension to compute the distance. Defaults to -1.

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return self.pb.dist(x.double(), y.double(), dim=dim).cuda(x.device)

    def poincare_distance_origin(self, x, dim=-1):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return self.pb.dist0(x.double(), dim=dim).cuda(x.device)

    def cosine_distance(self, x, y):
        """Cosine distance between two points.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)


class HyperMLR(nn.Module):
    """Multinomial logistic regression in hyperbolic space."""

    def __init__(self, out_channels, num_classes, c=1.0):
        """Initialize the model.

        Args:
            num_classes (int): Number of classes
            out_channels (int): Number of channels of the input features
            c (float, optional): Hyperbolic curvature. Defaults to 1.
        """
        super().__init__()
        self.c = c
        self.K = torch.tensor(c, dtype=float)
        self.num_classes = num_classes
        self.P_MLR = Parameter(
            torch.empty((num_classes, out_channels), dtype=torch.double)
        )
        self.A_MLR = Parameter(
            torch.empty((num_classes, out_channels), dtype=torch.double)
        )
        kaiming_uniform_(self.P_MLR, a=math.sqrt(5))
        kaiming_uniform_(self.A_MLR, a=math.sqrt(5))

    def _hyper_logits(self, inputs):
        """Compute the logits in hyperbolic space.

        Args:
            inputs (torch.Tensor): Tensor of shape (B, C, H, W)
        """
        # B = batch size
        # C = number of channels
        # H, W = height and width of the input
        # O = number of classes

        # P_MLR: (O,C)
        # A_MLR: (O,C)
        # output: (B,H,W,O)

        # normalize inputs and P_MLR
        xx = torch.norm(inputs, dim=1) ** 2  # (B,H,W)
        pp = torch.norm(-self.P_MLR, dim=1) ** 2  # (O,)
        P_kernel = -self.P_MLR[:, :, None, None]  # (O,C,1,1)

        # compute cross correlations
        px = torch.nn.functional.conv2d(
            input=inputs,
            weight=P_kernel,
            stride=1,
            padding="same",
            dilation=1,
            groups=1,
        )  # (B,O,H,W)
        pp = pp.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # (1,O,1,1)

        # c^2 * | X|^2 * |-P|^2
        sqsq = self.K * xx.unsqueeze(1) * self.K * pp  # (B,O,H,W)

        # Rewrite mob add as alpha * p + beta * x
        # where alpha = A / D and beta = B / D
        A = 1 + 2 * self.K * px + self.K * xx.unsqueeze(1)  # (B,O,H,W)
        B = 1 - self.K * pp  # (1,O,1,1)
        D = 1 + 2 * self.K * px + sqsq  # (B,O,H,W)
        D = torch.max(D, torch.tensor(1e-12, device=inputs.device))
        alpha = A / D  # (B,O,H,W)
        beta = B / D  # (B,O,H,W)

        # Calculate mobius addition norm indepently from the mobius addition
        # (B,O,H,W)
        mobaddnorm = (
            (alpha**2 * pp) + (beta**2 * xx.unsqueeze(1)) + (2 * alpha * beta * px)
        )
        # now in order to project the mobius addition onto the hyperbolic disc
        # we need to divide vectors whos l2norm : |x| (not |x|^2) are higher than max norm
        maxnorm = (1.0 - PROJ_EPS) / torch.sqrt(self.K)
        project_normalized = torch.where(  # (B,O,H,W)
            torch.sqrt(mobaddnorm) > maxnorm,  # condition
            maxnorm
            / torch.max(
                torch.sqrt(mobaddnorm), torch.tensor(1e-12, device=inputs.device)
            ),  # if true
            torch.ones_like(mobaddnorm),
        )  # if false
        mobaddnormprojected = torch.where(  # (B,O,H,W)
            torch.sqrt(mobaddnorm) < maxnorm,  # condition
            mobaddnorm,  # if true
            torch.ones_like(mobaddnorm) * maxnorm**2,
        )  # if false

        A_norm = torch.norm(self.A_MLR, dim=1)  # (O,)
        normed_A = torch.nn.functional.normalize(self.A_MLR, dim=1)  # (O,C)
        A_kernel = normed_A[:, :, None, None]  # (O,C,1,1)
        xdota = beta * torch.nn.functional.conv2d(inputs, weight=A_kernel)  # (B,O,H,W)
        pdota = (
            alpha * torch.sum(-self.P_MLR * normed_A, dim=1)[None, :, None, None]
        )  # (B,O,H,W)
        mobdota = xdota + pdota  # (B,O,H,W)
        mobdota *= project_normalized  # equiv to project mob add to max norm before dot
        lamb_px = 2.0 / torch.max(
            1 - self.K * mobaddnormprojected, torch.tensor(1e-12, device=inputs.device)
        )
        sineterm = torch.sqrt(self.K) * mobdota * lamb_px
        lambda_term = 2.0  # / (1 - self.K * pp)  # (1,O,1,1)
        out = (
            lambda_term
            / torch.sqrt(self.K)
            * A_norm.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            * torch.asinh(sineterm)
        )  # (B,O,H,W)
        return out

    def forward(self, x):
        logits = self._hyper_logits(x)
        return logits


class HyperMetrics(object):
    """Compute metrics for embeddings in euclidean and hyperbolic space.

    Args:
        c (float, optional): Hyperbolic curvature. Defaults to 1.0
    """

    def __init__(self, c=1.0) -> None:
        self.c = c
        self.mapper = HyperMapper(c=self.c)

    def compute(self, x, y):
        metrics = {}

        # MSE and Cosine distance
        metrics["mse"] = F.mse_loss(x, y)
        metrics["cosine_dist"] = self.mapper.cosine_distance(x, y)

        # Project embeddings to hyperbolic space
        x_h = self.mapper.expmap(x)
        y_h = self.mapper.expmap(y)

        # Radii in hyperbolic space
        radius_x = torch.linalg.norm(x_h, dim=-1)
        radius_y = torch.linalg.norm(y_h, dim=-1)
        metrics["radius_x"] = radius_x
        metrics["radius_y"] = radius_y

        # Angle between embeddings
        x_norm_e = x_h / radius_x.reshape(-1, 1)
        y_norm_e = y_h / radius_y.reshape(-1, 1)
        ang_e = torch.acos((x_norm_e * y_norm_e).sum(dim=-1)) * 180 / np.pi
        metrics["ang_e"] = ang_e

        # Poincare distance between embeddings
        metrics["poincare_dist"] = self.mapper.poincare_distance(x_h, y_h)

        return metrics
