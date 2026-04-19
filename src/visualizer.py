import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

def _compute_density(values, bins=50, value_range=None):
    if len(values) == 0:
        return None, None
    hist, edges = np.histogram(values, bins=bins, range=value_range, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, hist

def plot_training_history(history, output_path):
    """
    history: dict with keys 'train_loss', 'val_loss', 'avg_pearson', 'per_mark_pearson'
    """
    epochs = np.arange(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Loss Plot
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', marker='o', alpha=0.7)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', marker='s', alpha=0.7)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Pearson Correlation Plot
    axes[1].plot(epochs, history['avg_pearson'], label='Mean Pearson', color='black', linewidth=3, marker='D')
    for mark in config.TARGET_MARKS:
        if mark in history['per_mark_pearson']:
            axes[1].plot(epochs, history['per_mark_pearson'][mark], label=mark, alpha=0.6, linestyle='--')
            
    axes[1].set_title('Validation Pearson Correlation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pearson r')
    axes[1].set_ylim(-0.1, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_prediction_scatter(all_preds, all_targets, all_masks, output_path, max_points=5000):
    """
    Scatter plots of predictions vs targets for each mark.
    """
    num_marks = len(config.TARGET_MARKS)
    fig, axes = plt.subplots(1, num_marks, figsize=(6 * num_marks, 5.5))
    if num_marks == 1:
        axes = [axes]
        
    for m in range(num_marks):
        mark_name = config.TARGET_MARKS[m]
        valid_idx = all_masks[:, m] > 0.5
        
        p = all_preds[valid_idx, m]
        t = all_targets[valid_idx, m]
        
        if len(p) > max_points:
            idx = np.random.choice(len(p), max_points, replace=False)
            p, t = p[idx], t[idx]
            
        ax = axes[m]
        ax.scatter(t, p, alpha=0.2, s=5)
        
        # Perfect prediction line
        lo = min(t.min(), p.min())
        hi = max(t.max(), p.max())
        ax.plot([lo, hi], [lo, hi], 'r--', alpha=0.8)
        
        # Calculate local stats for the title
        corr = np.corrcoef(p, t)[0, 1] if len(p) > 1 else 0
        
        ax.set_title(f"{mark_name}\nPearson r: {corr:.4f}")
        ax.set_xlabel('True (Z-scored)')
        ax.set_ylabel('Predicted (Z-scored)')
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_correlation_bars(all_preds, all_targets, all_masks, output_path):
    """Bar chart of per-mark Pearson correlation for a quick visual summary."""
    marks = config.TARGET_MARKS
    corrs = []
    colors = []
    color_map = {
        'H3K27ac': '#e74c3c', 'H3K4me3': '#2ecc71', 'H3K27me3': '#3498db',
        'H3K9me3': '#9b59b6', 'H3K4me1': '#f39c12',
    }

    for m in range(len(marks)):
        valid = all_masks[:, m] > 0.5
        if valid.sum() > 2:
            corr = np.corrcoef(all_preds[valid, m], all_targets[valid, m])[0, 1]
            corrs.append(corr if not np.isnan(corr) else 0)
        else:
            corrs.append(0)
        colors.append(color_map.get(marks[m], '#95a5a6'))

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(marks, corrs, color=colors, edgecolor='white', linewidth=1.2)
    for bar, c in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{c:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Pearson r')
    ax.set_title('Per-Mark Validation Correlation')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_residual_distributions(all_preds, all_targets, all_masks, output_path):
    """Histogram of residuals (pred - target) per mark to spot bias."""
    marks = config.TARGET_MARKS
    num_marks = len(marks)
    fig, axes = plt.subplots(1, num_marks, figsize=(5 * num_marks, 4.5))
    if num_marks == 1:
        axes = [axes]

    for m in range(num_marks):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 2:
            continue
        residuals = all_preds[valid, m] - all_targets[valid, m]
        ax = axes[m]
        ax.hist(residuals, bins=60, color='steelblue', alpha=0.8, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', alpha=0.8)
        ax.axvline(residuals.mean(), color='orange', linestyle='-', alpha=0.8,
                   label=f'mean={residuals.mean():.3f}')
        ax.set_title(f'{marks[m]}')
        ax.set_xlabel('Residual (pred − true)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    fig.suptitle('Residual Distributions (Z-scored)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_val_signal_distributions(all_preds, all_targets, all_masks,
                                  norm_mean, norm_std, output_path):
    """
    Distribution overlays of actual vs predicted signal per mark,
    un-normalized back to log2(signal + 1) space — same style as
    the held-out evaluation plots but generated every epoch.
    """
    marks = config.TARGET_MARKS
    num_marks = len(marks)
    fig, axes = plt.subplots(2, num_marks, figsize=(6 * num_marks, 10))
    if num_marks == 1:
        axes = axes.reshape(2, 1)

    for m in range(num_marks):
        valid = all_masks[:, m] > 0.5
        if valid.sum() < 2:
            continue

        p_z = all_preds[valid, m]
        t_z = all_targets[valid, m]

        # Un-normalize
        p_raw = p_z * norm_std[m] + norm_mean[m]
        t_raw = t_z * norm_std[m] + norm_mean[m]

        value_range = (
            min(float(t_raw.min()), float(p_raw.min())),
            max(float(t_raw.max()), float(p_raw.max())) + 0.1,
        )

        # Top row: all samples overlay
        ax = axes[0, m]
        ax.hist(t_raw, bins=50, range=value_range, density=True,
                color='grey', alpha=0.55, label='Actual')
        ax.hist(p_raw, bins=50, range=value_range, density=True,
                color='coral', alpha=0.55, label='Predicted')
        ax.set_title(f'{marks[m]} — All Samples')
        ax.set_xlabel('log2(signal + 1)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # Bottom row: negatives vs positives predicted distributions
        is_positive = t_raw > 0
        ax = axes[1, m]
        if (~is_positive).sum() > 0:
            ax.hist(p_raw[~is_positive], bins=50, range=value_range, density=True,
                    color='royalblue', alpha=0.5, label='Predicted (actual=0)')
        if is_positive.sum() > 0:
            ax.hist(p_raw[is_positive], bins=50, range=value_range, density=True,
                    color='darkorange', alpha=0.5, label='Predicted (actual>0)')
        ax.set_title(f'{marks[m]} — Neg vs Pos')
        ax.set_xlabel('log2(signal + 1)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle('Validation Signal Distributions', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_epoch_visualizations(epoch, history, preds, targets, masks, output_dir,
                              norm_mean=None, norm_std=None):
    """Convenience wrapper to save all plots for an epoch."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Update history plot (global — overwritten each epoch)
    plot_training_history(history, os.path.join(output_dir, "training_history.png"))

    # 2. Per-epoch plots
    epoch_plot_dir = os.path.join(output_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_plot_dir, exist_ok=True)

    plot_prediction_scatter(
        preds, targets, masks,
        os.path.join(epoch_plot_dir, "val_scatter.png"))

    plot_correlation_bars(
        preds, targets, masks,
        os.path.join(epoch_plot_dir, "correlation_bars.png"))

    plot_residual_distributions(
        preds, targets, masks,
        os.path.join(epoch_plot_dir, "residual_distributions.png"))

    # 3. Signal distribution overlays (needs norm stats)
    if norm_mean is not None and norm_std is not None:
        plot_val_signal_distributions(
            preds, targets, masks, norm_mean, norm_std,
            os.path.join(epoch_plot_dir, "signal_distributions.png"))

def plot_heldout_distributions(pred_records, output_dir, bins=50):
    """
    pred_records: pandas DataFrame-like object with columns:
      mark, actual, predicted, present_true
    Saves one 2x2 distribution figure per mark.
    """
    os.makedirs(output_dir, exist_ok=True)

    for mark_name in config.TARGET_MARKS:
        mark_df = pred_records[pred_records["mark"] == mark_name]
        if len(mark_df) == 0:
            continue

        negatives = mark_df[mark_df["present_true"] == 0]
        positives = mark_df[mark_df["present_true"] == 1]

        all_vals = np.concatenate([
            mark_df["actual"].to_numpy(dtype=np.float32),
            mark_df["predicted"].to_numpy(dtype=np.float32),
        ])
        lo = float(np.min(all_vals))
        hi = float(np.max(all_vals))
        if lo == hi:
            hi = lo + 1.0
        value_range = (lo, hi)

        density_max = 0.0
        for values in [
            negatives["predicted"].to_numpy(dtype=np.float32),
            positives["predicted"].to_numpy(dtype=np.float32),
            positives["actual"].to_numpy(dtype=np.float32),
            mark_df["actual"].to_numpy(dtype=np.float32),
            mark_df["predicted"].to_numpy(dtype=np.float32),
        ]:
            _, dens = _compute_density(values, bins=bins, value_range=value_range)
            if dens is not None and len(dens) > 0:
                density_max = max(density_max, float(np.max(dens)))
        density_max = max(density_max * 1.05, 0.1)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Held-out Signal Distributions: {mark_name}", fontsize=16)

        ax = axes[0, 0]
        if len(negatives) > 0:
            ax.hist(
                negatives["predicted"].to_numpy(dtype=np.float32),
                bins=bins,
                range=value_range,
                density=True,
                color="coral",
                alpha=0.8,
                label="Predicted",
            )
        ax.set_title("Predicted for Negative Samples")
        ax.set_xlabel("log2(signal + 1)")
        ax.set_ylabel("Density")
        ax.set_ylim(0, density_max)
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[0, 1]
        if len(positives) > 0:
            ax.hist(
                positives["actual"].to_numpy(dtype=np.float32),
                bins=bins,
                range=value_range,
                density=True,
                color="grey",
                alpha=0.55,
                label="Actual",
            )
            ax.hist(
                positives["predicted"].to_numpy(dtype=np.float32),
                bins=bins,
                range=value_range,
                density=True,
                color="coral",
                alpha=0.55,
                label="Predicted",
            )
        ax.set_title("Positive Samples")
        ax.set_xlabel("log2(signal + 1)")
        ax.set_ylabel("Density")
        ax.set_ylim(0, density_max)
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[1, 0]
        if len(negatives) > 0:
            ax.hist(
                negatives["predicted"].to_numpy(dtype=np.float32),
                bins=bins,
                range=value_range,
                density=True,
                color="royalblue",
                alpha=0.5,
                label="Predicted, actual=0",
            )
        if len(positives) > 0:
            ax.hist(
                positives["predicted"].to_numpy(dtype=np.float32),
                bins=bins,
                range=value_range,
                density=True,
                color="darkorange",
                alpha=0.5,
                label="Predicted, actual>0",
            )
        ax.set_title("Predicted Negatives vs Positives")
        ax.set_xlabel("log2(signal + 1)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

        ax = axes[1, 1]
        ax.hist(
            mark_df["actual"].to_numpy(dtype=np.float32),
            bins=bins,
            range=value_range,
            density=True,
            color="grey",
            alpha=0.55,
            label="Actual",
        )
        ax.hist(
            mark_df["predicted"].to_numpy(dtype=np.float32),
            bins=bins,
            range=value_range,
            density=True,
            color="sandybrown",
            alpha=0.65,
            label="Predicted",
        )
        ax.set_title("All Held-out Samples")
        ax.set_xlabel("log2(signal + 1)")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"heldout_distribution_{mark_name}.png"), dpi=200)
        plt.close(fig)
