from ppm.wandb_utils import fetch_experiments


def read_args():
    import argparse

    parser = argparse.ArgumentParser(description="Fetch experiments from wandb")
    parser.add_argument("--project", type=str, default="", help="Project name")
    return parser.parse_args()


if __name__ == "__main__":
    args = read_args()
    print("Fetching experiments for project:", args.project)
    fetch_experiments(project=args.project)
    print("Done!")
