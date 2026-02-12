import argparse
import os
import datasets

def create_prompt(prompt_text, system_prompt=None):
    """
    Create chat-style prompt.
    """
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ]
    else:
        return [
            {"role": "user", "content": prompt_text},
        ]


def split_prompt_chosen_rejected(chosen_raw, rejected_raw):
    # split into lines
    chosen_lines = chosen_raw.strip().splitlines()
    rejected_lines = rejected_raw.strip().splitlines()

    # find the first index where they differ
    for i, (c, r) in enumerate(zip(chosen_lines, rejected_lines)):
        if c != r:
            split_idx = i
            break
    else:
        split_idx = min(len(chosen_lines), len(rejected_lines))

    # prompt = lines up to divergence
    prompt = "\n".join(chosen_lines[:split_idx])
    chosen_cont = "\n".join(chosen_lines[split_idx:])
    rejected_cont = "\n".join(rejected_lines[split_idx:])

    return prompt, chosen_cont, rejected_cont


def make_map_fn(split, params):
    """
    Returns a DPO-style example:
    {
        prompt: [...],
        chosen: "...",
        rejected: "...",
        split: "...",
        index: int
    }
    """
    def process_fn(example, idx):
        chosen_raw = example["chosen"]
        rejected_raw = example["rejected"]

        # Extract prompt + responses
        prompt_chosen, chosen_resp, rejected_resp = split_prompt_chosen_rejected(chosen_raw, rejected_raw)

        data = {
            "prompt": create_prompt(prompt_chosen, params.system_prompt),
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "split": split,
            "index": idx,
        }
        return data

    return process_fn


def create_file_name(params, split):
    fpart = "wsp" if params.system_prompt else "ns"
    return f"hh_rlhf_dpo_{params.run_id}_{fpart}_{split}.parquet"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="Anthropic/hh-rlhf")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    ########
    # load dataset
    ########
    dataset = datasets.load_dataset(args.data_source)
    full_train = dataset["train"]

    ########
    # create train/val/test split
    ########
    test_split = full_train.train_test_split(test_size=args.test_ratio, seed=args.seed)
    remaining = test_split["train"]
    test_dataset = test_split["test"]

    val_split = remaining.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = val_split["train"]
    val_dataset = val_split["test"]

    ########
    # map dataset to DPO format
    ########
    train_dataset = train_dataset.map(
        make_map_fn("train", args),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        make_map_fn("val", args),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=val_dataset.column_names,
    )

    test_dataset = test_dataset.map(
        make_map_fn("test", args),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=test_dataset.column_names,
    )

    ########
    # save datasets
    ########
    os.makedirs(args.local_dir, exist_ok=True)

    train_file = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file   = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file  = os.path.join(args.local_dir, create_file_name(args, "test"))

    train_dataset.to_parquet(train_file)
    val_dataset.to_parquet(val_file)
    test_dataset.to_parquet(test_file)

    ########
    # print a sample
    ########
    sample = train_dataset[0]
    print(40 * "=", "Sample", 40 * "=")
    print("Prompt:")
    print(sample["prompt"])
    print(80 * "=")
    print("Chosen:")
    print(sample["chosen"])
    print(80 * "=")
    print("Rejected:")
    print(sample["rejected"])

    print(f"Train file: {train_file} ({len(train_dataset)} examples)")
    print(f"Val file:   {val_file} ({len(val_dataset)} examples)")
    print(f"Test file:  {test_file} ({len(test_dataset)} examples)")
    print("Done.")
