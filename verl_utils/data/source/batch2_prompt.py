# w/o tool
system_prompt = """You are a software expert. You will be given a software issue and some patch candidates in user query. You need to judge which patch(es) can resolve the issue. Carefully review, critic, and compare the given candidates. You need to first think about the reasoning process in the mind until you get the final answer. Finally, put the ID(s) of correct patch candidates within \\boxed{}, e.g., \\boxed{1}, \\boxed{2, 4}, \\boxed{1, 2, 3, 4} (all correct), \\boxed{} (all wrong).
""".strip()

def process_fn(example, idx):

    alpha = 0.5
    id = example.pop("instance_id") # unique id
    repo = example.pop("repo") # repo name
    issue = example.pop("problem_statement") # original issue
    patch_list = example.pop("patch") # patches
    resolved = example.pop("resolved") # true or false
    sha = example.pop("base_commit") # sha of the base commit
    split = example.pop("split") # train, test
    loc = example.pop("oracle_location") # loc

    data = {
        "data_source": f'batch_{split}',
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"[Issue]\n{issue.strip()}\n\n"
                    f"[Patch 1]\n{patch_list[0].strip()}\n\n"
                    f"[Patch 2]\n{patch_list[1].strip()}\n\n"
                    f"[Patch 3]\n{patch_list[2].strip()}\n\n"
                    f"[Patch 4]\n{patch_list[3].strip()}\n\n"
                    "Which patch(es) can resolve the issue?"
                ),
            },
        ],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": resolved},
        "extra_info": {
            "split": split,
            "index": idx,
            "id": id,
            "repo": repo,
            "sha": sha,
            "alpha": alpha,
            "loc": loc,
            # "need_tools_kwargs": True, # for multi-turn
            # "tools_kwargs": {
            #     "get_code_of_methods": {
            #         "create_kwargs": {"issue_id": id},
            #     },
            # },
        },
    }
    return data