import os, re, subprocess, json
from collections import defaultdict
from tqdm import tqdm
import ast
import argparse
import shutil
from tempfile import TemporaryDirectory

from pathlib import Path
from unidiff import PatchSet

from datasets import load_dataset, DatasetDict, Dataset
from feabench.collect.utils import Repo, create_instance, add_issues_for_django
from feabench.collect.make_instructions import COLUMN_TO_PROMPT, PROMPT_FUNCTIONS, get_wrapped_request_content, get_wrapped_discussion_content, convert_files_map_into_list, convert_components_map_into_list, get_wrapped_definitions


GIT_TOKEN = os.environ.get("GITHUB_TOKEN")


def run_command(command, cwd=None, check=True):
    """
    Run a shell command and capture the output. Only print stderr on failure.
    
    :param command: A list of strings representing the command to be executed.
    :param cwd: The directory in which to execute the command.
    :param check: If True, raise an exception if the command exits with a non-zero status.
    :return: A CompletedProcess instance.
    """
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}:")
        print("Standard Output:", e.stdout)
        print("Standard Error:", e.stderr)
        raise  # Re-raise the exception after printing error messages


def get_default_branch(repo_testbed_dir):
    """
    Determine the default branch of the remote repository.
    
    :param repo_testbed_dir: The directory where the repository is cloned.
    :return: Name of the default branch.
    """
    result = subprocess.run(
        ['git', 'remote', 'show', 'origin'],
        cwd=repo_testbed_dir,
        capture_output=True,
        text=True,
        check=True
    )
    for line in result.stdout.splitlines():
        if 'HEAD branch:' in line:
            return line.split()[-1]
    return 'main'  # Fallback to 'main' if not found


def clone_or_update_repo(repo, repo_testbed_dir):
    """
    Clone the repository if it does not exist or update it to ensure it's a complete Git repository.
    
    :param repo: The name of the repository in the format 'owner/repo'.
    :param repo_testbed_dir: The directory where the repository should be cloned.
    """
    if os.path.exists(repo_testbed_dir) and os.path.isdir(repo_testbed_dir):
        # Check if it's a git repository
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--is-inside-work-tree'],
                cwd=repo_testbed_dir,
                capture_output=True,
                text=True,
                check=True
            )
            if 'true' in result.stdout.strip():
                print(f"Repository already exists at {repo_testbed_dir}, updating...")
                
                # Ensure the working directory is clean
                subprocess.run(['git', 'reset', '--hard', 'HEAD'], cwd=repo_testbed_dir, check=True)
                subprocess.run(['git', 'clean', '-fdx'], cwd=repo_testbed_dir, check=True)
                
                # Fetch all changes from the remote
                subprocess.run(['git', 'fetch', '--all'], cwd=repo_testbed_dir, check=True)
                
                # Determine the default branch
                default_branch = get_default_branch(repo_testbed_dir)
                print(f"Default branch detected as '{default_branch}'")
                
                # Switch to the default branch
                try:
                    subprocess.run(['git', 'checkout', default_branch], cwd=repo_testbed_dir, check=True)
                except subprocess.CalledProcessError:
                    # If the default branch doesn't exist locally, create it tracking the remote one
                    subprocess.run(['git', 'checkout', '-b', default_branch, f'origin/{default_branch}'], cwd=repo_testbed_dir, check=True)
                
                # Pull the latest changes from the remote default branch
                subprocess.run(['git', 'pull', '--ff-only', 'origin', default_branch], cwd=repo_testbed_dir, check=True)
                
                return
        except subprocess.CalledProcessError:
            pass  # Not a valid git repository, proceed with cloning
    
    print(f"Cloning repository {repo} into {repo_testbed_dir}...")
    # Remove any existing directory with the same name to avoid conflicts
    if os.path.exists(repo_testbed_dir):
        shutil.rmtree(repo_testbed_dir)
    
    # Clone the repository
    subprocess.run(['git', 'clone', f'https://github.com/{repo}.git', repo_testbed_dir], check=True)



def remove_comments_and_whitespace(line):
    """Remove Python comments and trailing whitespace from a line of code."""
    # Remove inline comments using regex
    cleaned_line = re.sub(r'#.*', '', line)
    # Strip trailing whitespace
    return cleaned_line.rstrip()


def apply_patch(repo_dir, patch_content):
    """Apply the patch content to the repository."""
    with open(os.path.join(repo_dir, 'patch.diff'), 'w') as f:
        f.write(patch_content)
    run_command(['git', 'apply', 'patch.diff'], cwd=repo_dir)
    os.remove(os.path.join(repo_dir, 'patch.diff'))


def parse_namespaces(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source = file.read()

    tree = ast.parse(source, filename=file_path)
    namespaces = []

    class NamespaceVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_namespace = []
            self.namespace_stack = []

        def visit_FunctionDef(self, node):
            self._handle_namespace(node, 'function')

        def visit_ClassDef(self, node):
            self._handle_namespace(node, 'class')

        def _handle_namespace(self, node, type_):
            start_line = node.lineno
            end_line = self._find_end_line(node)
            # Find the end of the definition (the line containing ':')
            end_def_line = self._find_definition_end_line(node)
            
            # Get all lines from start_line to end_def_line
            lines = source.splitlines()[start_line - 1:end_def_line]

            # Join the lines and remove comments and whitespace
            signature = ' '.join(line.strip() for line in lines)
            signature = remove_comments_and_whitespace(signature)

            # if end_def_line - start_line + 1 > 1:
            #     print(signature)

            namespace = {
                'type': type_,
                'name': '.'.join(self.current_namespace + [node.name]),
                'lines': (start_line, end_line),
                'signature': signature,
                'doc': ast.get_docstring(node) or ''
            }
            namespaces.append(namespace)

            # Enter new namespace
            self.namespace_stack.append(node.name)
            self.current_namespace.append(node.name)

            # Visit child nodes
            self.generic_visit(node)

            # Exit namespace
            self.current_namespace.pop()
            self.namespace_stack.pop()

        def _find_end_line(self, node):
            max_line = node.lineno
            for child_node in ast.walk(node):
                if hasattr(child_node, 'lineno'):
                    max_line = max(max_line, child_node.lineno)
            return max_line

        def _find_definition_end_line(self, node):
            """Find the line number where the function/class definition ends (the line containing ':')."""
            # Iterate through the node's body to find the first statement's line number
            if node.body:
                first_statement_line = node.body[0].lineno
                # The definition ends at the line before the first statement
                return first_statement_line - 1
            else:
                # If there's no body, the definition ends at the node's line
                return node.lineno
            

    visitor = NamespaceVisitor()
    visitor.visit(tree)

    return namespaces


def compare_namespaces(old_file_path, new_file_path):
    old_namespaces = parse_namespaces(old_file_path) if old_file_path else []
    new_namespaces = parse_namespaces(new_file_path)

    old_namespaces_dict = {ns['name']: ns for ns in old_namespaces}
    new_namespaces_dict = {ns['name']: ns for ns in new_namespaces}

    added_namespaces = []

    for name, ns in new_namespaces_dict.items():
        if name not in old_namespaces_dict:
            added_namespaces.append(ns)

    return added_namespaces


def handle_new_file(file_path):
    """Handle the case where the file is entirely new."""
    return parse_namespaces(file_path)


def checkout_to_commit(repo_dir, commit_sha):
    """Checkout the repository to a specific commit."""
    run_command(['git', 'checkout', '--quiet', commit_sha], cwd=repo_dir)


def get_file_related_info(instance, testbed_dir):
    patch = instance["patch"]
    base_commit_sha = instance["base_commit"]

    testbed_dir = Path(testbed_dir)

    # Checkout to the base commit
    checkout_to_commit(testbed_dir, base_commit_sha)

    # get readme files
    files = os.listdir(testbed_dir)
    files = list(filter(lambda x: os.path.isfile(x), files))
    readme_files = list(filter(lambda x: x.lower().startswith("readme"), files))
    instance["readmes"] = {}
    for readme_file in readme_files:
        with open(readme_file) as f:
            content = f.read()
        instance["readmes"][readme_file] = content

    # Save the original files' content before applying the patch
    instance["files"] = {}
    instance["non_py_patch"] = ""
    for hunk in PatchSet(patch):
        file_path = hunk.path
        full_file_path = testbed_dir / file_path
        if os.path.exists(full_file_path):
            instance["files"][file_path] = full_file_path.read_text()
        else:
            instance["files"][file_path] = None
        if not file_path.endswith((".py")):
            instance["non_py_patch"] += str(hunk)

    # # settings of instance
    # instance["context_type"] = "oracle"

    # Apply the patch
    apply_patch(testbed_dir, patch)

    # Get the info for "new_components"
    new_components = defaultdict(list)
    for file_path, original_content in instance["files"].items():
        if not file_path.endswith('.py'):
            continue  # Skip non-Python files
        
        new_full_file_path = testbed_dir / file_path

        if original_content is None and new_full_file_path.exists():
            # Handle new files: all components are considered new
            added_namespaces = handle_new_file(new_full_file_path)
        elif original_content is not None and new_full_file_path.exists():
            # Compare existing files to find new components
            with TemporaryDirectory() as temp_dir:
                old_file_path = Path(temp_dir) / 'old.py'
                new_file_path = Path(temp_dir) / 'new.py'
                
                old_file_path.write_text(original_content)
                new_file_path.write_text(new_full_file_path.read_text())

                added_namespaces = compare_namespaces(str(old_file_path), str(new_file_path))
        else:
            continue  # Skip non-existent files after applying the patch

        if added_namespaces:
            new_components[file_path].extend(added_namespaces)

    instance['new_components'] = dict(new_components)

    if "problem_statement" in instance:
        instance.pop("problem_statement")
    if "hints_text" in instance:
        instance.pop("hints_text")

    # Clean up after processing each instance
    run_command(['git', 'reset', '--hard', 'HEAD'], cwd=testbed_dir)
    run_command(['git', 'clean', '-fdx'], cwd=testbed_dir)



def get_source_data(data_item):
    repo = data_item["repo"]
    pull_number = data_item["pull_number"]
    repo_api = Repo(repo.split('/')[0], repo.split('/')[1], token=GIT_TOKEN)
    pull = repo_api.get_pull(pull_number)
    setattr(pull, "resolved_issues", repo_api.extract_resolved_issues(pull))
    instance = create_instance(repo_api, pull)
    add_issues_for_django(instance)
    return instance


def extract_statement_from_natural_prompt(natural_prompt):
    # ensure <request> and </request> appear only once
    request_start = natural_prompt.find("<request>")
    request_end = natural_prompt.find("</request>")
    request_content = natural_prompt[request_start + len("<request>"):request_end].strip()
    if natural_prompt.count("<request>") != 1 or natural_prompt.count("</request>") != 1:
        print("Request tag appears more than once. Request content extracted as below:")
        print(request_content)
    # ensure <definitions> and </definitions> appear only once
    natural_prompt = natural_prompt.replace("between the <definitions> and </definitions>", " ")
    definitions_start = natural_prompt.find("<definitions>")
    definitions_end = natural_prompt.find("</definitions>")
    definitions_content = natural_prompt[definitions_start + len("<definitions>"):definitions_end].strip()
    if natural_prompt.count("<definitions>") != 1 or natural_prompt.count("</definitions>") != 1:
        print("Definitions tag appears more than once. Definitions content extracted as below:")
        print(definitions_content)

    problem_statement = [
        "This is a feature request which requires a new feature to add in the code repository.",
        "<<NEW FEATURE REQUEST>>",
        request_content,
        "\n",
        "There are several new functions or classes that need to be implemented, using the definitions below: ",
        "<<NEW DEFINITIONS>>",
        definitions_content,
        "\n",
        "Please note that in addition to the newly added components mentioned above, you also need to make other code changes to ensure that the new feature can be executed properly.",
        "<<END>>"
    ]
    return "\n".join(problem_statement)

    

def save_dataset(instances, instruction_data_save_dir, standard_data_save_dir, lite_instruction_data_save_dir, lite_standard_data_save_dir, lite_ids):
    ## sorted by time
    instances = sorted(instances, key=lambda x: x['created_at'], reverse=True)

    ## Get prompt
    for instance in instances:
        for column in COLUMN_TO_PROMPT:
            instance[column] = PROMPT_FUNCTIONS[COLUMN_TO_PROMPT[column]](instance)

        instance["pull_request_text"] = get_wrapped_request_content(instance)
        instance["issue_text"] = get_wrapped_discussion_content(instance)

        if "problem_info" in instance:
            instance.pop("problem_info")

        definition_text = get_wrapped_definitions(instance, with_doc=True)

        # process new_components and files into Sequence that can be saved as Dataset
        instance["files"] = convert_files_map_into_list(instance["files"])
        instance["readmes"] = convert_files_map_into_list(instance["readmes"])
        instance["new_components"] = convert_components_map_into_list(instance["new_components"])

        problem_statement = [
            "This is a feature request which requires a new feature to add in the code repository.",
            "<<NEW FEATURE REQUEST>>",
            instance["pull_request_text"],
            "\n",
            "There are several new functions or classes that need to be implemented, using the definitions below: ",
            "<<NEW DEFINITIONS>>",
            definition_text,
            "\n",
            "Please note that in addition to the newly added components mentioned above, you also need to make other code changes to ensure that the new feature can be executed properly.",
            "<<END>>"
        ]
        problem_statement = "\n".join(problem_statement)
        instance["problem_statement"] = problem_statement
        instance["hints_text"] = instance["issue_text"]

    
    instruction_columns = [
        "instance_id",
        "pull_number",
        "repo",
        "version",
        "base_commit",
        "created_at",
        "patch",
        "test_patch",
        "non_py_patch",
        "new_components",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "pull_request_text", "issue_text",
        'readmes', 'files',
        "environment_setup_commit"
    ] + list(COLUMN_TO_PROMPT.keys())
    instruction_dataset = [{key: item[key] for key in instruction_columns} for item in instances]
    # reserved_dataset = instances
    instruction_instances = Dataset.from_list(instruction_dataset)
    new_dataset = DatasetDict({
        "test": instruction_instances
    })

    new_dataset.save_to_disk(instruction_data_save_dir)
    print(f"Prompted dataset ({len(instruction_instances)}) saved to: {instruction_data_save_dir}")

    lite_instruction_instances = instruction_instances.filter(lambda ins: ins["instance_id"] in lite_ids)
    new_dataset = DatasetDict({
        "test": lite_instruction_instances
    })

    new_dataset.save_to_disk(lite_instruction_data_save_dir)
    print(f"Lite prompted dataset ({len(lite_instruction_instances)}) saved to: {lite_instruction_data_save_dir}")


    standard_columns = [
        "instance_id",
        "pull_number",
        "repo",
        "version",
        "base_commit",
        "created_at",
        "patch",
        "test_patch",
        "non_py_patch",
        "new_components",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "problem_statement", "hints_text",
        "environment_setup_commit"
    ]
    standard_dataset = [{key: item[key] for key in standard_columns} for item in instances]
    # reserved_dataset = instances
    standard_instances = Dataset.from_list(standard_dataset)
    new_dataset = DatasetDict({
        "test": standard_instances
    })

    new_dataset.save_to_disk(standard_data_save_dir)
    print(f"Standard dataset ({len(standard_instances)}) saved to: {standard_data_save_dir}")

    lite_standard_instances = standard_instances.filter(lambda ins: ins["instance_id"] in lite_ids)
    new_dataset = DatasetDict({
        "test": lite_standard_instances
    })

    new_dataset.save_to_disk(lite_standard_data_save_dir)
    print(f"Lite prompted dataset ({len(lite_standard_instances)}) saved to: {lite_standard_data_save_dir}")


def load_existing_data(file_path: str):
    """
    Load existing data from a JSONL file and return a dictionary with instance_id as keys.
    """
    existing_data = {}
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["instance_id"]] = instance
    return existing_data

def save_instance_to_file(instance, file_path: str):
    """
    Save a single instance to a JSONL file.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(instance, f, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to original FEA-Bench dataset", required=True)
    parser.add_argument("--lite_ids", type=str, help="Path to instance ids list file", required=True)
    parser.add_argument("--medium_file", type=str, help="Path to medium file with sufficient information.", required=True)
    parser.add_argument("--standard_dataset_path", type=str, help="Path to the standard dataset without any instructions.", required=True)
    parser.add_argument("--oracle_dataset_path", type=str, help="Path to the oracle dataset with instructions.", required=True)
    parser.add_argument("--lite_standard_dataset_path", type=str, help="Path to the standard dataset without any instructions (lite version).", required=True)
    parser.add_argument("--lite_oracle_dataset_path", type=str, help="Path to the oracle dataset with instructions (lite version).", required=True)
    parser.add_argument("--testbed", type=str, help="Path to testbed directory", required=True)
    args = parser.parse_args()

    # Load existing data from medium_file
    existing_instances = load_existing_data(args.medium_file)

    data = load_dataset(args.dataset)["test"]
    source_task_instances = []

    # f = open("/home/v-weili8/FEA-Bench/feabench-data/FEA-Bench-v1.0-medium.jsonl2", "w")

    for item in tqdm(data, desc="Get PR info"):
        instance_id = item["instance_id"]
        if instance_id in existing_instances:
            # print(f"Skipping instance {instance_id} as it already exists in the medium_file.")
            source_task_instances.append(existing_instances[instance_id])
            # # instance with information from dataset
            # existing_instances[instance_id]["version"] = item["version"]
            # existing_instances[instance_id]["FAIL_TO_PASS"] = item["FAIL_TO_PASS"]
            # existing_instances[instance_id]["PASS_TO_PASS"] = item["PASS_TO_PASS"]
            # existing_instances[instance_id]["environment_setup_commit"] = item["environment_setup_commit"]
            # f.write(json.dumps(existing_instances[instance_id], ensure_ascii=False) + "\n")
            continue

        repo_folder = item["repo"].replace('/', '__')
        repo_testbed = os.path.join(args.testbed, repo_folder)
        clone_or_update_repo(item["repo"], repo_testbed)
        instance = get_source_data(item)
        os.makedirs(args.testbed, exist_ok=True)
        get_file_related_info(instance, testbed_dir=repo_testbed)
        
        # instance with information from dataset
        instance["version"] = item["version"]
        instance["FAIL_TO_PASS"] = item["FAIL_TO_PASS"]
        instance["PASS_TO_PASS"] = item["PASS_TO_PASS"]
        instance["environment_setup_commit"] = item["environment_setup_commit"]

        if instance["instance_id"].startswith("open-edge-platform__"):
            instance["instance_id"] = instance["instance_id"].replace("open-edge-platform", "openvinotoolkit")
            instance["repo"] = instance["repo"].replace("open-edge-platform", "openvinotoolkit")

        source_task_instances.append(instance)

        # Save the new instance to medium_file
        save_instance_to_file(instance, args.medium_file)

    # load lite version of FEA-Bench
    lite_ids = json.load(open(args.lite_ids))

    save_dataset(
        instances=source_task_instances,
        instruction_data_save_dir=args.oracle_dataset_path,
        standard_data_save_dir=args.standard_dataset_path,
        lite_instruction_data_save_dir=args.lite_oracle_dataset_path,
        lite_standard_data_save_dir=args.lite_standard_dataset_path,
        lite_ids=lite_ids
    )

