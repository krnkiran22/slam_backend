"""
Compute RPE (Relative Pose Error) over 3-minute windows.

Primary metric for this project — NOT ATE.

Usage:
    python eval/compute_rpe.py --est estimated.txt --ref groundtruth.txt
"""
import argparse
import logging

logger = logging.getLogger(__name__)


def compute_rpe(estimated_path: str, reference_path: str, delta_s: float = 180.0) -> dict:
    """
    Compute RPE with a configurable time delta (default 180s = 3 minutes).
    Returns dict with rmse, mean, median, std, min, max.
    """
    from evo.tools import file_interface
    from evo.core import metrics, sync

    traj_ref = file_interface.read_tum_trajectory_file(reference_path)
    traj_est = file_interface.read_tum_trajectory_file(estimated_path)
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

    rpe = metrics.RPE(
        pose_relation=metrics.PoseRelation.translation_part,
        delta=delta_s,
        delta_unit=metrics.Unit.seconds,
    )
    rpe.process_data((traj_ref, traj_est))

    stats = {
        "rmse": rpe.get_statistic(metrics.StatisticsType.rmse),
        "mean": rpe.get_statistic(metrics.StatisticsType.mean),
        "median": rpe.get_statistic(metrics.StatisticsType.median),
        "std": rpe.get_statistic(metrics.StatisticsType.std),
        "min": rpe.get_statistic(metrics.StatisticsType.min),
        "max": rpe.get_statistic(metrics.StatisticsType.max),
    }
    return stats


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute RPE over 3-minute windows")
    parser.add_argument("--est", required=True, help="Estimated trajectory (TUM format)")
    parser.add_argument("--ref", required=True, help="Ground truth trajectory (TUM format)")
    parser.add_argument("--delta", type=float, default=180.0, help="Window size in seconds")
    args = parser.parse_args()

    stats = compute_rpe(args.est, args.ref, args.delta)

    print(f"\nRPE Results (delta={args.delta}s):")
    print(f"  RMSE:   {stats['rmse']:.6f}")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Median: {stats['median']:.6f}")
    print(f"  Std:    {stats['std']:.6f}")
    print(f"  Min:    {stats['min']:.6f}")
    print(f"  Max:    {stats['max']:.6f}")


if __name__ == "__main__":
    main()
