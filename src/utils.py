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
    df_with_guest = df.filter(~pl.col("Guest_Popularity_percentage").is_null())
    df_without_guest = df.filter(pl.col("Guest_Popularity_percentage").is_null())

    groups_with_guest = df_with_guest.group_by(
        ["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Day"]
    )
    unique_groups_with_guest = groups_with_guest

    groups_without_guest = df_without_guest.group_by(["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Publication_Day"])
    unique_groups_without_guest = groups_without_guest
    breakpoint()

    # Apply KFold on both group types
    df = df.with_columns(pl.lit(-1).alias("fold"))
    kf = KFold(n_splits=cfg.num_fold, shuffle=True, random_state=42)
    with_guest_splits = list(kf.split(unique_groups_with_guest))
    without_guest_splits = list(kf.split(unique_groups_without_guest))

    # Create a mapping dictionary for faster lookups
    fold_mapping_with_guest = {}
    fold_mapping_without_guest = {}

    for i in range(cfg.num_fold):
        with_guest_index = with_guest_splits[i][1]
        without_guest_index = without_guest_splits[i][1]

        # Get fold validation groups efficiently
        valid_with_guest = unique_groups_with_guest.filter(pl.Series(range(len(unique_groups_with_guest))).is_in(with_guest_index))

        valid_without_guest = unique_groups_without_guest.filter(pl.Series(range(len(unique_groups_without_guest))).is_in(without_guest_index))

        # Add keys to mapping dictionary
        for row in valid_with_guest.to_dicts():
            key = (row["Podcast_Name"], row["Episode_Title"], row["Host_Popularity_percentage"], row["Guest_Popularity_percentage"], row["Publication_Day"])
            fold_mapping_with_guest[key] = i

        for row in valid_without_guest.to_dicts():
            key = (row["Podcast_Name"], row["Episode_Title"], row["Host_Popularity_percentage"], row["Publication_Day"])
            fold_mapping_without_guest[key] = i

    # Apply folds in one go using expressions
    # First handle with_guest cases
    df_with_folds = df.with_columns(
        [
            pl.when(~pl.col("Guest_Popularity_percentage").is_null())
            .then(
                pl.struct(["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Guest_Popularity_percentage", "Publication_Day"]).apply(
                    lambda x: fold_mapping_with_guest.get(
                        (x["Podcast_Name"], x["Episode_Title"], x["Host_Popularity_percentage"], x["Guest_Popularity_percentage"], x["Publication_Day"]), -1
                    )
                )
            )
            .when(pl.col("Guest_Popularity_percentage").is_null())
            .then(
                pl.struct(["Podcast_Name", "Episode_Title", "Host_Popularity_percentage", "Publication_Day"]).apply(
                    lambda x: fold_mapping_without_guest.get((x["Podcast_Name"], x["Episode_Title"], x["Host_Popularity_percentage"], x["Publication_Day"]), -1)
                )
            )
            .otherwise(pl.col("fold"))
            .alias("fold")
        ]
    )

    return df_with_folds
