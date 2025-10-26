from typing import Tuple

import torch


class WhiteningTransform:
    """
    Manages whitening transformations for key and query vectors in attention mechanism.

    This class computes and applies whitening transforms to normalize the distribution
    of projected keys and queries, which can improve the effectiveness of approximate
    nearest neighbor search (ANNS) for attention computation.
    """

    def __init__(self):
        """Initialize whitening transform with None values."""
        self.mu_k = None              # Mean vectors: (khead, head_dim)
        self.L_inv = None             # Whitening transform for keys: (khead, head_dim, head_dim)
        self.L_inv_T = None           # Whitening transform for queries: (khead, head_dim, head_dim)
        self.mu_k_expanded = None     # Expanded mean for queries: (1, num_attention_heads, 1, head_dim)
        self.L_inv_T_expanded = None  # Expanded transform for queries: (1, num_attention_heads, head_dim, head_dim)

    @staticmethod
    @torch.no_grad()
    def _compute_whitener_psd(
        K: torch.Tensor,
        center: bool = True,
        keep_variance: float = 1.0,   # e.g., 0.98 to drop tiny eigens
        kappa_target: float = 30.0,   # cap condition number of transformed space
        alpha: float = 1.0,           # 1.0=full whitening, 0.5=power whitening
        diag_only: bool = False,      # True: standardize by variances only
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, T) where routing uses  K_w = (K - mu) @ T, Q_w = (Q - mu) @ T.

        T is symmetric PSD (ZCA-style):  T = V diag( (lambda')^{-alpha/2} ) V^T
          - alpha in [0,1]: 1 => full whitening; 0.5 => power-whitening.
          - We floor eigenvalues to enforce condition number <= kappa_target.
          - If keep_variance < 1, we drop tail eigenvectors to reach that cumulative
            variance and T projects onto the kept subspace (rank reduction).
          - If diag_only, we ignore off-diagonals: T = diag( (var_i')^{-alpha/2} ).
        """
        assert K.ndim == 2
        N, r = K.shape
        if center:
            mu = K.mean(dim=0)
            Kc = K - mu
        else:
            mu = torch.zeros(r, device=K.device, dtype=K.dtype)
            Kc = K

        # Covariance (full or diagonal)
        if diag_only:
            var = Kc.var(dim=0, unbiased=True).clamp_min(1e-12)
            lam = var
            V = torch.eye(r, device=K.device, dtype=K.dtype)
        else:
            cov = (Kc.T @ Kc) / max(1, N - 1)
            # Symmetric eigendecomposition (ascending eigenvalues)
            lam, V = torch.linalg.eigh(cov.float())

        # Sort descending for variance accounting
        idx_desc = torch.argsort(lam, descending=True)
        lam = lam[idx_desc]
        V = V[:, idx_desc]

        # Optional PCA truncation by explained variance
        if keep_variance < 1.0:
            cum = torch.cumsum(lam, dim=0) / (lam.sum() + 1e-12)
            m = int((cum <= keep_variance).sum().item()) + 1
            m = max(1, min(m, lam.numel()))
            lam = lam[:m]
            V = V[:, :m]
            r_eff = m
        else:
            r_eff = lam.numel()

        lam_max = lam.max().item()
        lam_min = lam.min().item()
        # Floor eigenvalues to control condition number: lam' = max(lam, floor)
        if kappa_target is not None and kappa_target > 1.0:
            lam_floor = max(lam_max / kappa_target, 1e-12)
            lam = torch.clamp(lam, min=lam_floor)
        lam_pow = lam.pow(-alpha * 0.5)  # exponent -alpha/2

        # Build symmetric transform T in full ambient dimension (r x r)
        T = V @ torch.diag(lam_pow) @ V.T
        # If we reduced rank, components orthogonal to span(V) are zeroed (projection)
        return mu, T

    def compute_partial_whitening(self, phi_k_samples, shrink=1e-3, partial_whitening=0.1):
        """
        Compute robust partial whitening transformation using Cholesky decomposition.

        Args:
            phi_k_samples: Tensor of shape (num_key_value_heads, num_samples, head_dim)
                          Contains φ_k(X) = X @ Q_k for multiple X samples
            shrink: Tikhonov regularization parameter for numerical stability
            partial_whitening: Interpolation parameter [0, 1]. 0 = no whitening, 1 = full whitening
        """
        num_key_value_heads, num_samples, head_dim = phi_k_samples.shape
        mu_list = []
        L_inv_list = []

        for head in range(num_key_value_heads):
            # Get samples for this head: (num_samples, head_dim)
            K = phi_k_samples[head]  # (N, r)

            # Compute whitener using the robust PSD approach
            mu, Linv_full = self._compute_whitener_psd(K, center=True, kappa_target=30.0, alpha=1.0)

            if partial_whitening == 0.0:
                # No whitening - return identity
                Linv_partial = torch.eye(head_dim, device=K.device, dtype=K.dtype)
            else:
                # Interpolate between identity and full whitening
                identity = torch.eye(head_dim, device=K.device, dtype=K.dtype)
                Linv_partial = (1 - partial_whitening) * identity + partial_whitening * Linv_full

            mu_list.append(mu)
            L_inv_list.append(Linv_partial)

        self.mu_k = torch.stack(mu_list, dim=0)  # (num_key_value_heads, head_dim)
        self.L_inv = torch.stack(L_inv_list, dim=0)  # (num_key_value_heads, head_dim, head_dim)
        self.L_inv_T = self.L_inv.transpose(-2, -1)  # (num_key_value_heads, head_dim, head_dim)

    def compute_adaptive_whitening(self, phi_k_samples, target_recall=0.9):
        """
        Compute adaptive whitening that preserves ranking by limiting the whitening strength.
        Uses robust PSD-safe whitening with conservative parameters.

        Args:
            phi_k_samples: Tensor of shape (num_key_value_heads, num_samples, head_dim)
            target_recall: Target recall to maintain (default: 0.9)
        """
        num_key_value_heads, num_samples, head_dim = phi_k_samples.shape
        mu_list = []
        T_list = []

        for head in range(num_key_value_heads):
            # Get samples for this head: (num_samples, head_dim)
            K = phi_k_samples[head]  # (N, r)

            # Compute whitener using conservative PSD parameters to preserve ranking
            mu, T = self._compute_whitener_psd(
                K,
                center=True,
                keep_variance=0.98,    # Keep all variance (no PCA truncation)
                kappa_target=30.0,    # Moderate condition number control
                alpha=0.1,           # Only 10% whitening to preserve ranking
                diag_only=False
            )

            mu_list.append(mu)
            T_list.append(T)

        self.mu_k = torch.stack(mu_list, dim=0)  # (num_key_value_heads, head_dim)
        self.L_inv = torch.stack(T_list, dim=0)   # (num_key_value_heads, head_dim, head_dim)
        self.L_inv_T = self.L_inv  # T is symmetric, so T^T = T

    def apply_whitening_to_keys(self, embed_keys):
        """
        Apply whitening transformation to embedded keys.

        Args:
            embed_keys: Tensor of shape (batch_size, khead, seq_len, head_dim)

        Returns:
            embed_keys_whitened: Whitened keys k_tilde = (phi_k - mu) @ L^{-1}
        """
        if self.mu_k is None or self.L_inv is None:
            raise ValueError("Must call compute_adaptive_whitening or estimate_and_initialize first")

        mu_k_expanded = self.mu_k.unsqueeze(0).unsqueeze(2)  # (1, khead, 1, head_dim)
        embed_keys_centered = embed_keys.float() - mu_k_expanded
        embed_keys_whitened = torch.matmul(embed_keys_centered, self.L_inv)
        return embed_keys_whitened

    def apply_whitening_to_queries(self, embed_queries):
        """
        Apply whitening transformation to embedded queries.

        Args:
            embed_queries: Tensor of shape (batch_size, num_attention_heads, seq_len, head_dim)

        Returns:
            embed_queries_whitened: Whitened queries q_tilde = (phi_q - mu) @ L^{-T}
        """
        if self.mu_k_expanded is None or self.L_inv_T_expanded is None:
            raise ValueError("Must call estimate_and_initialize first to expand transforms for queries")

        embed_queries_centered = embed_queries - self.mu_k_expanded
        embed_queries_whitened = torch.matmul(embed_queries_centered, self.L_inv_T_expanded)
        return embed_queries_whitened

    @torch.no_grad()
    def estimate_and_initialize(self, phi_k: torch.Tensor, num_key_value_groups: int):
        """
        Estimate covariance from samples and initialize whitening transforms.

        Args:
            phi_k: Tensor of shape (khead, num_samples, head_dim)
                   Representative input samples for covariance estimation
            num_key_value_groups: Number of times to repeat key head transforms for query heads
        """
        khead, num_samples, head_dim = phi_k.shape

        # Compute whitening transforms with conservative partial whitening
        self.compute_adaptive_whitening(phi_k)

        # Expand for query heads (repeat key head transforms for each query head group)
        L_inv_T_expanded = self.L_inv_T.repeat_interleave(num_key_value_groups, dim=0)  # (num_attention_heads, head_dim, head_dim)
        mu_k_expanded = self.mu_k.repeat_interleave(num_key_value_groups, dim=0)  # (num_attention_heads, head_dim)
        self.mu_k_expanded = mu_k_expanded.unsqueeze(0).unsqueeze(2)  # (1, num_attention_heads, 1, head_dim)
        self.L_inv_T_expanded = L_inv_T_expanded.unsqueeze(0)  # (1, num_attention_heads, head_dim, head_dim)

        print(f"Estimated covariance for {khead} heads using {num_samples} samples")