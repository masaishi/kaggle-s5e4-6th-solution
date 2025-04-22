import git
import polars as pl
from sklearn.model_selection import KFold

from config import cfg


def commit_results(val_score, wandb_run_name):
    try:
        repo = git.Repo(search_parent_directories=True)
        # Add all changes
        repo.git.add(".")
        commit_message = f"Val score: {val_score:.6f} | wandb run: {wandb_run_name}"
        repo.git.commit("-m", commit_message)

        commit_id = repo.head.object.hexsha
        commit_message = repo.head.object.message.strip()
        branch_name = repo.active_branch.name

        return {
            "commit_id": commit_id,
            "commit_message": commit_message,
            "branch_name": branch_name,
        }
    except Exception as e:
        print(f"Error during git commit: {e}")
        return {
            "commit_id": "unknown",
            "commit_message": f"Failed commit with val score: {val_score:.6f}",
            "branch_name": "unknown",
        }


def get_index_splits(df, cfg=cfg):
    # Filter out rows with null Guest_Popularity_percentage
    df_with_guest = df.filter(~pl.col("Guest_Popularity_percentage").is_null())
    df_without_guest = df.filter(pl.col("Guest_Popularity_percentage").is_null())

    groups_with_guest = df_with_guest.group_by(
        ["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Day"]
    )
    unique_groups_with_guest = groups_with_guest.agg(pl.count()).drop("count")

    groups_without_guest = df_without_guest.group_by(["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Publication_Day"])
    unique_groups_without_guest = groups_without_guest.agg(pl.count()).drop("count")

    # Apply KFold on both group types
    kf = KFold(n_splits=cfg["num_fold"], shuffle=True, random_state=42)
    with_guest_splits = list(kf.split(unique_groups_with_guest))
    without_guest_splits = list(kf.split(unique_groups_without_guest))

    # Concate both splits to make splits
    index_splits = []
    for i in range(cfg["num_fold"]):
        with_guest_index = with_guest_splits[i][1]
        without_guest_index = without_guest_splits[i][1]

        index_splits.append([with_guest_index.tolist() + without_guest_index.tolist(), with_guest_splits[i][0].tolist() + without_guest_splits[i][0].tolist()])
    return index_splits
