import git

import wandb


class WandbCallback:
    def __init__(self, log_every=50, log_feature_importance=True):
        self.log_every = log_every
        self.iteration = 0
        self.log_feature_importance = log_feature_importance

    def __call__(self, env):
        # This gets called after each iteration
        if self.iteration % self.log_every == 0:
            metrics = {}
            for dataset_name, eval_name, value, _ in env.evaluation_result_list:
                metric_name = f"{dataset_name}/{eval_name}"
                metrics[metric_name] = value

            # Log the metrics to wandb
            if self.log_feature_importance and hasattr(env, "model"):
                feature_names = env.model.feature_name()
                importance = env.model.feature_importance(importance_type="split")

                feature_importance = {name: imp for name, imp in zip(feature_names, importance)}

                wandb.log(
                    {
                        "feature_importance": wandb.Table(
                            data=[
                                [k, v]
                                for k, v in sorted(
                                    feature_importance.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )
                            ],
                            columns=["feature", "importance"],
                        )
                    },
                    step=self.iteration,
                )

                wandb.log(
                    {
                        "feature_importance_plot": wandb.plot.bar(
                            wandb.Table(
                                data=[
                                    [k, v]
                                    for k, v in sorted(
                                        feature_importance.items(),
                                        key=lambda x: x[1],
                                        reverse=True,
                                    )[:10]
                                ],
                                columns=["feature", "importance"],
                            ),
                            "feature",
                            "importance",
                            title="Top Feature Importance",
                        )
                    },
                    step=self.iteration,
                )

            wandb.log(metrics, step=self.iteration)

        self.iteration += 1
        return False


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
