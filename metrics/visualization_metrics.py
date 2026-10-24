"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

-----------------------------

visualization_metrics.py

Note: Use PCA or tSNE for generated and original data visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualization(ori_data, generated_data, analysis, save_path=None, seed=42):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data (shape: [N, seq_len, dim])
      - generated_data: generated synthetic data (shape: [M, seq_len, dim])
      - analysis: 'tsne' or 'pca'
      - save_path: optional file path to save the figure
      - seed: random seed for reproducible subsampling and t-SNE
    """
    ori_arr = np.asarray(ori_data, dtype=np.float32)
    gen_arr = np.asarray(generated_data, dtype=np.float32)

    anal_sample_no = min([1000, len(ori_arr), len(gen_arr)])
    rng = np.random.default_rng(seed)

    idx_ori = rng.permutation(len(ori_arr))[:anal_sample_no]
    idx_gen = rng.permutation(len(gen_arr))[:anal_sample_no]

    ori_sub = ori_arr[idx_ori]
    gen_sub = gen_arr[idx_gen]

    # Vectorized mean across feature dimension (axis=2) -> shape (anal_sample_no, seq_len)
    prep_data = np.mean(ori_sub, axis=2)
    prep_data_hat = np.mean(gen_sub, axis=2)

    colors = ["red"] * anal_sample_no + ["blue"] * anal_sample_no

    if analysis == "pca":
        pca = PCA(n_components=2, random_state=seed)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        f, ax = plt.subplots(1)
        plt.scatter(
            pca_results[:, 0],
            pca_results[:, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            pca_hat_results[:, 0],
            pca_hat_results[:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()
        plt.title("PCA plot")
        plt.xlabel("x-pca")
        plt.ylabel("y-pca")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()

    elif analysis == "tsne":
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)
        eff_perplexity = min(40, max(5, len(prep_data_final) // 3 - 1))

        tsne = TSNE(
            n_components=2,
            verbose=0,
            perplexity=eff_perplexity,
            max_iter=1000,
            random_state=seed,
        )
        tsne_results = tsne.fit_transform(prep_data_final)

        f, ax = plt.subplots(1)
        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()
        plt.title("t-SNE plot")
        plt.xlabel("x-tsne")
        plt.ylabel("y-tsne")
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.show()
