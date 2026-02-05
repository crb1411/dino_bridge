# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import torch
import torch.distributed as torch_dist
import torch.nn as nn
import torch.nn.functional as F

import dinov3.distributed as dist


class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(
        self,
        gate_threshold: float = 0.0,
        gate_alpha: float = 0.1,
        gate_enabled: bool | None = False,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.gate_threshold = float(gate_threshold)
        self.gate_alpha = float(gate_alpha)
        self.gate_enabled = gate_enabled
        self.chunk_size = chunk_size

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, indices = torch.max(dots, dim=1)  # max inner prod -> min distance
        return indices

    def _apply_gate(self, losses: torch.Tensor) -> torch.Tensor:
        enabled = self.gate_enabled
        if not enabled:
            return losses.mean()
        alpha = max(self.gate_alpha, 1e-6)
        gate = torch.sigmoid((losses - self.gate_threshold) / alpha)
        weighted_mean = (losses * gate).mean()
        gated_mean = (losses * gate).sum() / gate.sum().clamp_min(1.0)
        return 0.5 * weighted_mean + 0.5 * gated_mean

    def _loss_on_batch(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)
            indices = self.pairwise_NNs_inner(student_output)
            distances = self.pdist(student_output, student_output[indices])  # BxD, BxD -> B
            losses = -torch.log(distances + eps)
            loss = self._apply_gate(losses)
        return loss

    def forward(self, student_output, eps=1e-8, chunk_size: int | None = None, shuffle: bool = True):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        if student_output.shape[0] < 2:
            return student_output.new_tensor(0.0)

        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_size is None or chunk_size >= student_output.shape[0] or chunk_size <= 0:
            return self._loss_on_batch(student_output, eps=eps)

        if shuffle:
            perm = torch.randperm(student_output.shape[0], device=student_output.device)
            student_output = student_output[perm]

        n_chunks = student_output.shape[0] // chunk_size
        if n_chunks <= 0:
            return student_output.new_tensor(0.0)
        student_output = student_output[: n_chunks * chunk_size]
        chunk_losses = []
        for start in range(0, student_output.shape[0], chunk_size):
            chunk = student_output[start : start + chunk_size]
            if chunk.shape[0] < 2:
                continue
            chunk_losses.append(self._loss_on_batch(chunk, eps=eps))
        if not chunk_losses:
            return student_output.new_tensor(0.0)
        return torch.stack(chunk_losses).mean()


class KoLeoLossDistributed(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    def __init__(
        self,
        topk=1,
        loss_group_size: int | None = None,
        gate_threshold: float = 0.0,
        gate_alpha: float = 0.1,
        gate_enabled: bool | None = None,
        chunk_size: int | None = None,
    ):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)
        self.topk = topk
        self.loss_group_size = loss_group_size  # Size of the nearest neighbor set. If None, uses global batch size.
        self.gate_threshold = float(gate_threshold)
        self.gate_alpha = float(gate_alpha)
        self.gate_enabled = gate_enabled
        self.chunk_size = chunk_size

    def pairwise_NNs_inner(self, x, all_x, rank):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, all_x.t())  # local_B x global_B
        local_B, global_B = dots.shape
        dots.view(-1)[rank * local_B :: (global_B + 1)].fill_(-1)  # Trick to fill diagonal with -1
        _, indices = torch.topk(dots, dim=1, k=self.topk)  # max inner prod -> min distance
        return indices

    def _apply_gate(self, losses: torch.Tensor) -> torch.Tensor:
        enabled = self.gate_enabled
        if enabled is None:
            enabled = self.gate_threshold > 0.0
        if not enabled:
            return losses.mean()
        alpha = max(self.gate_alpha, 1e-6)
        gate = torch.sigmoid((losses - self.gate_threshold) / alpha)
        weighted_mean = (losses * gate).mean()
        gated_mean = (losses * gate).sum() / gate.sum().clamp_min(1.0)
        return 0.5 * weighted_mean + 0.5 * gated_mean

    def _loss_on_batch(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.autocast("cuda", enabled=False):
            student_output = F.normalize(student_output, eps=eps, p=2, dim=-1)  # local_B x D

            if dist.is_enabled():
                all_student_outputs = torch.cat(torch_dist.nn.all_gather(student_output), dim=0)  # global_B x D
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                all_student_outputs = student_output
                world_size = 1
                rank = 0

            # Group the global batch into groups of size `loss_group_size` and use the features of the group
            # the local rank falls into as the nearest neighbor set for the local rank
            local_B = len(student_output)
            global_B = len(all_student_outputs)
            loss_group_size = self.loss_group_size if self.loss_group_size is not None else global_B
            if loss_group_size % local_B != 0:
                raise ValueError(
                    f"Loss group size size {loss_group_size} must be a multiple of local batch size {local_B}."
                )
            if global_B % loss_group_size != 0:
                raise ValueError(
                    f"Global batch size {global_B} must be divisible by loss group size {loss_group_size}."
                )
            n_groups = global_B // loss_group_size
            ranks_per_group = world_size // n_groups
            rank_in_group = rank % ranks_per_group
            group = rank // ranks_per_group
            all_student_outputs = all_student_outputs.view(n_groups, loss_group_size, student_output.shape[1])
            all_student_outputs = all_student_outputs[group]  # loss_group_size x D

            with torch.no_grad():
                indices = self.pairwise_NNs_inner(student_output, all_student_outputs, rank_in_group)  # local_B x topk

            student_output_expanded = (
                student_output.unsqueeze(1).repeat(1, self.topk, 1).flatten(0, 1)
            )  # (local_B * topk) x D
            distances = self.pdist(student_output_expanded, all_student_outputs[indices].flatten(0, 1))
            losses = -torch.log(distances.float() + eps)
            loss = self._apply_gate(losses)

        return loss

    def forward(self, student_output, eps=1e-8, chunk_size: int | None = None, shuffle: bool = False):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        if student_output.shape[0] < 2:
            return student_output.new_tensor(0.0)

        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_size is None or chunk_size >= student_output.shape[0] or chunk_size <= 0:
            return self._loss_on_batch(student_output, eps=eps)

        if shuffle:
            perm = torch.randperm(student_output.shape[0], device=student_output.device)
            student_output = student_output[perm]

        n_chunks = student_output.shape[0] // chunk_size
        if n_chunks <= 0:
            return student_output.new_tensor(0.0)
        student_output = student_output[: n_chunks * chunk_size]
        chunk_losses = []
        for start in range(0, student_output.shape[0], chunk_size):
            chunk = student_output[start : start + chunk_size]
            if chunk.shape[0] < 2:
                continue
            chunk_losses.append(self._loss_on_batch(chunk, eps=eps))
        if not chunk_losses:
            return student_output.new_tensor(0.0)
        return torch.stack(chunk_losses).mean()
