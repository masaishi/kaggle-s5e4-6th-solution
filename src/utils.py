import git


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
