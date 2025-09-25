import pandas as pd
import os
import re
import ast
import json
import argparse
import sys
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from verl_utils.data.utils.localize import get_hunk_location
from verl_utils.data.utils.classify import get_classified_instance_ids


def get_module_name(file_path, root_dir):
    """
    Transfer a file path to a module name.
    Args:
        file_path (str): The path of the file.
        root_dir (str): The root directory of the repo.
    Returns:
        str: The module name.
    """
    rel_path = os.path.relpath(file_path, root_dir)
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    return rel_path.replace(os.sep, ".")

def extract_functions_from_ast(source, tree, module_path):
    """
    Extract functions from a file's AST and return a dictionary of {"name": "code"}.
    Args:
        source (str): The source code of the file
        tree (ast.AST): The AST of the file.
        module_path (str): The path of the module.
    Returns:
        dict: A dictionary of {"name": "code"}.
    """
    class_stack = []
    functions_ast = {}
    def visit(node):
        if isinstance(node, ast.ClassDef):
            class_stack.append(node.name)
            for child in node.body:
                visit(child)
            class_stack.pop()
        elif isinstance(node, ast.FunctionDef):
            if class_stack:
                class_name = class_stack[-1]
                key = f"{module_path}.{class_name}.{node.name}"
            else:
                key = f"{module_path}.{node.name}"
            try:
                func_source = ast.get_source_segment(source, node)
                functions_ast[key] = {
                    "code": func_source,
                    "dependencies": [],
                    "dependents": []
                }
            except Exception as e:
                print(f"Error unparsing {key}: {e}")
        elif hasattr(node, 'body') and isinstance(node.body, list):
            for child in node.body:
                visit(child)
    visit(tree)
    return functions_ast

def find_method_calls(node):
    calls = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            func = n.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    return list(calls)

def get_func_ast(root_dir):
    """
    Extract functions from all python files AST in a repo and return a dictionary of {"name": "code"}.
    Args:
        root_dir (str): The root directory of the repo.
    Returns:
        functions_ast (dict): The mapping between function names and code.
    """
    func_ast = {}
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".py"):
                full_path = os.path.join(root, f)
                try:
                    source = open(full_path, "r", encoding='utf-8').read()
                    tree = ast.parse(source, filename=full_path)
                except Exception as e:
                    print(f"Error parsing {full_path}: {e}")
                    continue
                module_path = get_module_name(full_path, root_dir)
                func_ast.update(extract_functions_from_ast(source, tree, module_path))
    
    parsed_trees = {}
    for fqn in func_ast:
        try:
            parsed_trees[fqn] = ast.parse(func_ast[fqn]["code"])
        except SyntaxError:
            continue
    
    for fqn, tree_node in parsed_trees.items():
        calls = find_method_calls(tree_node)
        
        for call in calls:
            parts = fqn.split(".")
            
            if len(parts) >= 3:
                module_path = ".".join(parts[:-2])
                class_name = parts[-2]
                candidate = f"{module_path}.{class_name}.{call}"
                if candidate in func_ast and candidate != fqn:
                    if candidate not in func_ast[fqn]["dependencies"]:
                        func_ast[fqn]["dependencies"].append(candidate)
                    if fqn not in func_ast[candidate]["dependents"]:
                        func_ast[candidate]["dependents"].append(fqn)
                    continue
            
            if len(parts) > 1:
                module_path = ".".join(parts[:-1])
                candidate = f"{module_path}.{call}"
                if candidate in func_ast and candidate != fqn:
                    if candidate not in func_ast[fqn]["dependencies"]:
                        func_ast[fqn]["dependencies"].append(candidate)
                    if fqn not in func_ast[candidate]["dependents"]:
                        func_ast[candidate]["dependents"].append(fqn)
                    continue
            

            for key in func_ast:
                if key.endswith(f".{call}") and key != fqn:
                    if key not in func_ast[fqn]["dependencies"]:
                        func_ast[fqn]["dependencies"].append(key)
                    if fqn not in func_ast[key]["dependents"]:
                        func_ast[key]["dependents"].append(fqn)
                    break

    return func_ast

def get_pure_patch(patch_str):
    """
    Remove reproduce test patches in the model generated patch string.
    Args:
        patch_str (str): The patch string.
    Returns:
        filtered_patches (list): The filtered patch list.
    """
    patches = re.split(r'(?=^diff --git a/)', patch_str, flags=re.MULTILINE)
    filtered_patches = []
    for patch in patches:
        m = re.match(r'diff --git a/(.*?) b/(.*?)\n', patch)
        if not m:
            continue
        origin_filename = m.group(1)
        filename_lower = origin_filename.lower()
        if ".py" not in filename_lower \
            or "test" in filename_lower and 'pytest' not in filename_lower\
            or "reproduce" in filename_lower \
            or origin_filename == '/dev/null' \
            or re.search(r'^new file mode \d{6}$', patch, flags=re.MULTILINE) \
            or re.search(r'^--- /dev/null$', patch, flags=re.MULTILINE):
            continue
        filtered_patches.append(patch)
    return filtered_patches

def get_issue_info(path, type_classify='single'):
    """
    Get the basic info of the issue from the dataset.
    Args:
        path (str): The path of the dataset.
        type_classify (str): 'single', 'multiple', None
    Returns:
        df (pd.DataFrame): The basic info of the issue.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not exists.")
    raw_df = pd.read_parquet(path)
    df = raw_df[["instance_id", "repo", "problem_statement", "base_commit", "patch"]]

    # get the hunk location of golden patch
    df['oracle_location'] = df.apply(lambda row: get_hunk_location(row['patch']), axis=1)
    df = df[df['oracle_location'].str.len() > 0]
    print(f'Get {len(df)} issues with func-level edit from {len(raw_df)} issues.')

    # get the filtered instance ids
    if type_classify:
        ids_single, ids_multiple = get_classified_instance_ids(path)
        if type_classify == 'single':
            ids = ids_single
        elif type_classify == 'multiple':
            ids = ids_multiple
        else:
            raise ValueError(f" type_classify must be 'single', 'multiple' or 'all', but got {type_classify}")
        df = df[df['instance_id'].isin(ids)]
        df = df.reset_index(drop=True)
        print(f'Filtered {len(df)} issues from {len(raw_df)} issues.')
    else:
        print(f'Get {len(df)} issues from original issue dataset.')
    return df

def get_patch_info(path, split, trajs):
    """
    Get the patch info from a certain method's rollouts.
    Args:
        path (str): The path of the rollouts logs (including `patch.diff` and `report.json` in SWE-bench format, or including `output.swebench_eval.jsonl` in SWE-gym format).
        split (str): The split of the dataset.
    Returns:
        df (pd.DataFrame): The filtered patches without reproduce or tests.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not exists.")
    info = list()
    if split == 'train' or split == 'sft': # using swegym format
        path = os.path.join(path, 'output.swebench_eval.jsonl')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                temp_info = json.loads(line)
                patch_str = temp_info['test_result']['git_patch']
                filtered_patches = get_pure_patch(patch_str)
                if filtered_patches == []:
                    continue
                instance = dict()
                instance['instance_id'] = temp_info['instance_id']
                instance['resolved'] = temp_info['test_result']['report']['resolved']
                instance['patch'] = '\n'.join(filtered_patches)
                info.append(instance)
    elif split == 'test': # using swebench format
        for root, dirs, files in os.walk(path):
            for dir in dirs:
                patch_file = os.path.join(root, dir, "patch.diff")
                patch_str = open(patch_file, "r", encoding='utf-8').read().strip()
                filtered_patches = get_pure_patch(patch_str)
                
                report_file = os.path.join(root, dir, "report.json")
                if not os.path.exists(report_file):
                    continue
                report_dict = json.load(open(report_file, "r", encoding='utf-8'))
                resolved = report_dict[dir]["resolved"]

                if trajs:
                    if 'SA' in root:
                        traj_path = path.replace('logs', 'trajs') + f'/{dir}.traj'
                        if not os.path.exists(traj_path):
                            continue
                        with open(traj_path, 'r') as f:
                            trajectory = json.load(f)['history']
                    elif 'OH' in root:
                        traj_path = path.replace('logs', 'trajs') + f'/{dir}.json'
                        if not os.path.exists(traj_path):
                            continue
                        with open(traj_path, 'r') as f:
                            trajectory = json.load(f)
                    else:
                        raise NotImplementedError

                instance = dict()
                instance['instance_id'] = dir
                instance['resolved'] = resolved
                instance['patch'] = "\n".join(filtered_patches)
                instance['trajs'] = trajectory
                info.append(instance)

    df = pd.DataFrame(info)
    return df

def merge_basic_info(issue_info, patch_info, split):
    """
    Merge the basic info of the issue and the patch info.
    Args:
        issue_info (pd.DataFrame): The basic info.
        patch_info (pd.DataFrame): The patch info.
        split (str): The split of the dataset.
    Returns:
        df (pd.DataFrame): The merged info without empty filtered patches.
    """
    if split != 'oracle':
        df = issue_info.drop(columns=['patch'])
        df = df.merge(patch_info, on="instance_id", how="inner")
        df = df[df['patch'].str.len() > 0]
        df = df.reset_index(drop=True)
        print(f'Filtered {len(df)} non-empty patches from {len(patch_info)} rollouts after merged, total {len(issue_info)} issues.')
        return df
    else:
        df = issue_info
        df['resolved'] = True
        df['predicted_location'] = df['oracle_location']
        print(f'Get {len(df)} issues with oracle patch. Rollout patch is ignored without merge.')
        return df

def get_context(basic_df, path='data'):
    """
    Get additional context from git checkout repo and save full information to disk.
    Args:
        basic_df (pd.DataFrame): The merged basic info.
        path (str): The path to save the git environment and db.
    Returns:
        basic_df (pd.DataFrame): The merged basic info with additional context.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    os.makedirs(os.path.join(path, "cached_env"), exist_ok=True)

    db_path = os.path.join(path, 'context.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS context (
            instance_id TEXT PRIMARY KEY,
            func_ast TEXT
        )
    ''')
    conn.commit()

    for idx, row in basic_df.iterrows():
        instance_id = str(row["instance_id"])
        
        print(f'Processing instance_id: {instance_id}')

        cursor.execute('SELECT 1 FROM context WHERE instance_id = ?', (instance_id,))
        if cursor.fetchone() is not None:
            print(f"Instance ID {instance_id} AST already processed, skipping AST extraction.")
            continue

        repo = row["repo"]
        base_commit = row["base_commit"]
        temp_path = os.path.join(path, "cached_env", repo.replace("/", "__"))
        cmd = ""
        if not os.path.exists(temp_path):
            print(f'{repo} is not found in default {temp_path} directory')
            print(f"Cloning {repo} to {temp_path}")
            cmd = f"git clone git@github.com:{repo}.git {temp_path} && "
        cmd += f'cd {temp_path} && git checkout -f {base_commit}'
        os.system(cmd)
        func_ast = get_func_ast(temp_path)

        cursor.execute(
            'INSERT OR REPLACE INTO context (instance_id, func_ast) VALUES (?, ?)',
            (instance_id, json.dumps(func_ast))
        )
        conn.commit()

    conn.close()

    print(f"Processing finished and data saved to database {db_path}.")
    return basic_df

def db2dir(db_path):
    from tqdm import tqdm
    """
    Transfer the db to dir.
    To avoid read conflic when async rollout samples in verl.
    """
    save_path = db_path.replace('.db', '')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM context')
    rows = cursor.fetchall()

    for row in tqdm(rows):
        instance_id, func_ast = row
        func_ast = json.loads(func_ast)
        full_ast = {}
        for method, impl in func_ast.items():
            if len(method.split('.')) < 3:
                continue
            full_ast[method] = impl

            
        with open(os.path.join(save_path, f'{instance_id}.json'), 'w') as f:
            json.dump(full_ast, f)

    conn.close()
    print(f"Data saved to {save_path}.")


def get_info(issue_path, patch_path, save_path, type_classify, split, tag, trajs):
    issue_info = get_issue_info(issue_path, type_classify)
    patch_info = get_patch_info(patch_path, split, trajs)
    basic_df = merge_basic_info(issue_info, patch_info, split)
    basic_df['split'] = split
    basic_df.to_parquet(os.path.join(save_path, f"info_{split}_{tag}.parquet"))
    print(f'Get full info done, total {len(basic_df)} issues.')
    print(f'Basic Info Saved to {save_path}/info_{split}_{tag}.parquet')
    # print('Start to get AST contenxt ...')
    # basic_df = get_context(basic_df, save_path)
    # basic_df.to_parquet(os.path.join(save_path, f"info_{split}_{tag}.parquet"))
    # print(f'Full Info Saved to {save_path}/info_{split}_{tag}.parquet')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract the meta information from the dataset and rollouts.')
    parser.add_argument('--issue_path', type=str, default='data/datasets/swe-gym.parquet', help='Path to the issue info dataset.')
    parser.add_argument('--patch_path', type=str, default='data/rollouts/20240620_SA_claude3.5/logs', help='Path to the rollouts.')
    parser.add_argument('--save_path', type=str, default='data/', help='Path to save the full info.')
    parser.add_argument('--type_classify', type=str, default='', help='Classify of the issue type, e.g., single means only select issues invonving single file edit. Options: single, multiple, or None.')
    parser.add_argument('--split', type=str, default='train', help='Split of the dataset.')
    parser.add_argument('--tag', type=str, default='03', help='Special tag for info file name.')
    parser.add_argument('--oracle_mode', type=bool, default=False, help='Whether to use oracle mode, i.e., use the golden patch as the rollout.')
    parser.add_argument('--db2dir', type=bool, default=False, help='Whether to transfer the db to dir.')
    parser.add_argument('--trajs', type=bool, default=False, help='Whether to load trajectory in the meta data.')
    args = parser.parse_args()
    
    if not args.oracle_mode:
        split = args.split
    else:
        print('### WARNNING: Oracle mode is on. Rollout patch will be replaced with the golden patch. ###')
        split = 'oracle'

    if args.db2dir:
        db2dir(args.save_path + 'context.db')
    else:
        # get_info("data/datasets/swe-bench-verified.parquet", "data/rollouts/20240620_SA_claude3.5/logs", "data/", 'train')
        get_info(args.issue_path, args.patch_path, args.save_path, args.type_classify, split, args.tag, args.trajs)
