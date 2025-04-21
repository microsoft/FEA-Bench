from __future__ import annotations


import logging
import re
import requests
import time

from bs4 import BeautifulSoup
from ghapi.core import GhApi
from fastcore.net import HTTP404NotFoundError, HTTP403ForbiddenError
from typing import Callable, Iterator, Optional, Dict
import json
from unidiff import PatchSet

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Repo:
    def __init__(self, owner: str, name: str, token: Optional[str] = None):
        """
        Init to retrieve target repository and create ghapi tool

        Args:
            owner (str): owner of target repository
            name (str): name of target repository
            token (str): github token
        """
        self.owner = owner
        self.name = name
        self.token = token
        self.api = GhApi(token=token)
        self.repo = self.call_api(self.api.repos.get, owner=owner, repo=name)

    def call_api(self, func: Callable, **kwargs) -> dict|None:
        """
        API call wrapper with rate limit handling (checks every 5 minutes if rate limit is reset)

        Args:
            func (callable): API function to call
            **kwargs: keyword arguments to pass to API function
        Return:
            values (dict): response object of `func`
        """
        while True:
            try:
                values = func(**kwargs)
                return values
            except HTTP403ForbiddenError as e:
                while True:
                    rl = self.api.rate_limit.get()
                    logger.info(
                        f"[{self.owner}/{self.name}] Rate limit exceeded for token {self.token[:10]}, "
                        f"waiting for 5 minutes, remaining calls: {rl.resources.core.remaining}"
                    )
                    if rl.resources.core.remaining > 0:
                        break
                    time.sleep(60 * 5)
            except HTTP404NotFoundError as e:
                logger.info(f"[{self.owner}/{self.name}] Resource not found {kwargs}")
                return None

    def extract_resolved_issues(self, pull: dict) -> list[str]:
        """
        Extract list of issues referenced by a PR

        Args:
            pull (dict): PR dictionary object from GitHub
        Return:
            resolved_issues (list): list of issue numbers referenced by PR
        """
        # Define 1. issue number regex pattern 2. comment regex pattern 3. keywords
        issues_pat = re.compile(r"(\w+)\s+\#(\d+)")
        comments_pat = re.compile(r"(?s)<!--.*?-->")
        keywords = {
            "close",
            "closes",
            "closed",
            "fix",
            "fixes",
            "fixed",
            "resolve",
            "resolves",
            "resolved",
        }

        # Construct text to search over for issue numbers from PR body and commit messages
        text = pull.title if pull.title else ""
        text += "\n" + (pull.body if pull.body else "")
        commits = self.get_all_loop(
            self.api.pulls.list_commits, pull_number=pull.number, quiet=True
        )
        commit_messages = [commit.commit.message for commit in commits]
        commit_text = "\n".join(commit_messages) if commit_messages else ""
        text += "\n" + commit_text
        # Remove comments from text
        text = comments_pat.sub("", text)
        # Look for issue numbers in text via scraping <keyword, number> patterns
        references = dict(issues_pat.findall(text))
        resolved_issues = list()
        if references:
            for word, issue_num in references.items():
                if word.lower() in keywords:
                    resolved_issues.append(issue_num)
        return resolved_issues

    def get_all_loop(
        self,
        func: Callable,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        quiet: bool = False,
        **kwargs,
    ) -> Iterator:
        """
        Return all values from a paginated API endpoint.
        
        Args:
            func (callable): API function to call
            per_page (int): number of values to return per page
            num_pages (int): number of pages to return
            quiet (bool): whether to print progress
            **kwargs: keyword arguments to pass to API function
        """
        page = 1
        args = {
            "owner": self.owner,
            "repo": self.name,
            "per_page": per_page,
            **kwargs,
        }
        while True:
            try:
                # Get values from API call
                values = func(**args, page=page)
                yield from values
                if len(values) == 0:
                    break
                if not quiet:
                    rl = self.api.rate_limit.get()
                    logger.info(
                        f"[{self.owner}/{self.name}] Processed page {page} ({per_page} values per page). "
                        f"Remaining calls: {rl.resources.core.remaining}"
                    )
                if num_pages is not None and page >= num_pages:
                    break
                page += 1
            except Exception as e:
                # Rate limit handling
                logger.error(
                    f"[{self.owner}/{self.name}] Error processing page {page} "
                    f"w/ token {self.token[:10]} - {e}"
                )
                while True:
                    rl = self.api.rate_limit.get()
                    if rl.resources.core.remaining > 0:
                        break
                    logger.info(
                        f"[{self.owner}/{self.name}] Waiting for rate limit reset "
                        f"for token {self.token[:10]}, checking again in 5 minutes"
                    )
                    time.sleep(60 * 5)
        if not quiet:
            logger.info(
                f"[{self.owner}/{self.name}] Processed {(page-1)*per_page + len(values)} values"
            )

    def get_all_issues(
        self,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        direction: str = "desc",
        sort: str = "created",
        state: str = "closed",
        quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all issues from repo

        Args:
            per_page (int): number of issues to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort issues
            sort (str): field to sort issues by
            state (str): state of issues to look for
            quiet (bool): whether to print progress
        """
        issues = self.get_all_loop(
            self.api.issues.list_for_repo,
            num_pages=num_pages,
            per_page=per_page,
            direction=direction,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return issues

    def get_all_pulls(
        self,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        direction: str = "desc",
        sort: str = "created",
        state: str = "closed",
        quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all PRs from repo

        Args:
            per_page (int): number of PRs to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort PRs
            sort (str): field to sort PRs by
            state (str): state of PRs to look for
            quiet (bool): whether to print progress
        """
        pulls = self.get_all_loop(
            self.api.pulls.list,
            num_pages=num_pages,
            direction=direction,
            per_page=per_page,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return pulls

    def get_pull(self, pull_number: int) -> Dict:
        """
        Retrieve information about a specific pull request in the repository

        Args:
            pull_number (int): The number of the pull request

        Returns:
            Dict: A dictionary containing the pull request details
        """
        pr_details = self.call_api(self.api.pulls.get, owner=self.owner, repo=self.name, pull_number=pull_number)
        return pr_details


    def get_all_commits(
        self,
        per_page: int = 100,
        num_pages: Optional[int] = None,
        direction: str = "desc",
        sort: str = "created",
        state: str = "closed",
        quiet: bool = False,
    ) -> Iterator:
        """
        Wrapper for API call to get all PRs from repo

        Args:
            per_page (int): number of PRs to return per page
            num_pages (int): number of pages to return
            direction (str): direction to sort PRs
            sort (str): field to sort PRs by
            state (str): state of PRs to look for
            quiet (bool): whether to print progress
        """
        pulls = self.get_all_loop(
            self.api.repos.list_commits,
            num_pages=num_pages,
            direction=direction,
            per_page=per_page,
            sort=sort,
            state=state,
            quiet=quiet,
        )
        return pulls



def extract_problem_statement_and_hints(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Extract problem statement from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    if repo.name == "django":
        return extract_problem_statement_and_hints_django(pull, repo)
    text = ""
    all_hint_texts = list()
    for issue_number in pull["resolved_issues"]:
        issue = repo.call_api(
            repo.api.issues.get,
            owner=repo.owner,
            repo=repo.name,
            issue_number=issue_number,
        )
        if issue is None:
            continue
        title = issue.title if issue.title else ""
        body = issue.body if issue.body else ""
        text += f"{title}\n{body}\n"
        issue_number = issue.number
        hint_texts = _extract_hints(pull, repo, issue_number)
        hint_text = "\n".join(hint_texts)
        all_hint_texts.append(hint_text)
    return text, "\n".join(all_hint_texts) if all_hint_texts else ""


def extract_problem_info(pull: dict, repo: Repo):
    """
    Extract problem statement from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        info : all info about PR and issues
    """
    
    info = {
        "first_commit_time": -1,
        "pr_title": pull["title"],
        "pr_body": pull["body"],
        "pr_timeline": [],
        "issues": {}
    }

    ## Get first commit time
    # Get all commits in PR
    commits = repo.get_all_loop(
        repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
    )
    commits = list(commits)
    if len(commits) == 0:
        # If there are no comments, return no hints
        commit_time = -1
    else:
        # Get time of first commit in PR
        commit_time = commits[0].commit.author.date  # str
        commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))
    
    info["first_commit_time"] = commit_time

    ## Get all contents in PR
    all_contents = repo.get_all_loop(
        repo.api.issues.list_comments, issue_number=pull["number"], quiet=True
    )
    all_contents = list(all_contents)

    pr_comments = list()
    for content in all_contents:
        comment_time = time.mktime(
            time.strptime(content.updated_at, "%Y-%m-%dT%H:%M:%SZ")
        )  # use updated_at instead of created_at
        pr_comments.append({"time": comment_time, "comment": content.body})
        # all comment will be saved

    info["pr_timeline"] = pr_comments

    ## Get all issue contents
    for issue_number in pull["resolved_issues"]:
        issue = repo.call_api(
            repo.api.issues.get,
            owner=repo.owner,
            repo=repo.name,
            issue_number=issue_number,
        )
        if issue is None:
            continue
        title = issue.title if issue.title else ""
        body = issue.body if issue.body else ""
        issue_number = issue.number
        issue_comments = _extract_hints(commit_time, pull, repo, issue_number)
        
        info["issues"][issue_number] = {
            "issue_title": title,
            "issue_body": body,
            "issue_timeline": issue_comments
        }
    
    return info


def _extract_hints(first_commit_time: float, pull: dict, repo: Repo, issue_number: int) -> list[str]:
    """
    Extract hints from comments associated with a pull request (before first commit)

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
        issue_number (int): issue number
    Return:
        hints (list): list of hints
    """
    if first_commit_time < 0:
        return []

    # Get all comments in PR issue
    all_comments = repo.get_all_loop(
        repo.api.issues.list_comments, issue_number=issue_number, quiet=True
    )
    all_comments = list(all_comments)

    # Iterate through all comments, only keep comments created before first commit
    comments = list()
    for comment in all_comments:
        comment_time = time.mktime(
            time.strptime(comment.updated_at, "%Y-%m-%dT%H:%M:%SZ")
        )  # use updated_at instead of created_at
        comments.append({"time": comment_time, "comment": comment.body})
    # Keep text from comments
    return comments


def is_test_file(path):
    if any(
        test_word in path for test_word in
        ['test', 'tests', 'e2e', 'testing', 'check']
    ):
        if not any(path.endswith(ext) for ext in NON_TEST_EXTS):
            return True
    
    return False


def extract_patches(pull: dict, repo: Repo) -> tuple[str, str]:
    """
    Get patch and test patch from PR

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """
    patch = requests.get(pull["diff_url"]).text
    patch_test = ""
    patch_fix  = ""
    for hunk in PatchSet(patch):
        if any(
            test_word in hunk.path for test_word in
            ['test', 'tests', 'e2e', 'testing', 'check']
        ):
            if not any(hunk.path.endswith(ext) for ext in NON_TEST_EXTS):
                patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    return patch_fix, patch_test


def extract_repo_full_name(github_commit_url):
    url = github_commit_url.rstrip('/')
    
    path_parts = url.split('/')
    
    if len(path_parts) >= 5 and path_parts[2] == 'github.com':
        return f"{path_parts[3]}/{path_parts[4]}"
    else:
        raise ValueError("Invalid GitHub commit URL")


def extract_patches_from_commit(commit: dict, repo: Repo) -> tuple[str, str]:
    """
    Get patch and test patch from commit

    Args:
        commit (dict): commit dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        patch_change_str (str): gold patch
        patch_test_str (str): test patch
    """

    repo_name = extract_repo_full_name(commit["html_url"])
    diff_url = f"https://github.com/{repo_name}/commit/{commit['sha']}.diff"

    patch = requests.get(diff_url).text
    patch_test = ""
    patch_fix  = ""
    for hunk in PatchSet(patch):
        if any(
            test_word in hunk.path for test_word in
            ['test', 'tests', 'e2e', 'testing', 'check']
        ):
            patch_test += str(hunk)
        else:
            patch_fix += str(hunk)
    
    return patch_fix, patch_test


def extract_problem_info_from_commit(commit: dict, repo: Repo):

    commit_time = commit["commit"]["author"]["date"]  # str
    commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))


    return {
        "message": commit["commit"]["message"],
        "commit_time": commit_time,
    }


def add_issues_for_django(instance):
    if instance["repo"] == "django/django":
        for issue_no in instance["issue_numbers"]:
            # get issue discussions
            issue_content = {}
            url = f"https://code.djangoproject.com/ticket/{issue_no}"
            resp = requests.get(url)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")

            # Get problem statement (title + body)
            issue_desc = soup.find("div", {"id": "ticket"})
            title = issue_desc.find("h1", class_="searchable").get_text()
            title = re.sub(r"\s+", " ", title).strip()
            body = issue_desc.find("div", class_="description").get_text()
            body = re.sub(r"\n+", "\n", body)
            body = re.sub(r"    ", "\t", body)
            body = re.sub(r"[ ]{2,}", " ", body).strip()

            issue_content["issue_title"] = title
            issue_content["issue_body"] = body
            issue_content["issue_timeline"] = []

            # Get all comments
            comments_html = soup.find("div", {"id": "changelog"})
            div_blocks = comments_html.find_all("div", class_="change")
            # Loop through each div block
            for div_block in div_blocks:
                # Find the comment text and timestamp
                comment_resp = div_block.find("div", class_="comment")
                timestamp_resp = div_block.find("a", class_="timeline")
                if comment_resp is None or timestamp_resp is None:
                    continue

                comment_text = re.sub(r"\s+", " ", comment_resp.text).strip()
                timestamp = timestamp_resp["title"]
                if timestamp.startswith("See timeline at "):
                    timestamp = timestamp[len("See timeline at ") :]
                if "/" in timestamp:
                    timestamp = time.mktime(time.strptime(timestamp, "%m/%d/%y %H:%M:%S"))
                elif "," in timestamp:
                    timestamp = time.mktime(time.strptime(timestamp, "%b %d, %Y, %I:%M:%S %p"))
                else:
                    raise ValueError(f"Timestamp format not recognized: {timestamp}")
                
                issue_content["issue_timeline"].append(
                    {
                        "time": timestamp,
                        "comment": comment_text
                    }
                )
            
            # assign to instance inplace
            instance["problem_info"]["issues"][issue_no] = issue_content



### MARK: Repo Specific Parsing Functions ###
def extract_problem_statement_and_hints_django(
    pull: dict, repo: Repo
) -> tuple[str, list[str]]:
    """
    Get problem statement and hints from issues associated with a pull request

    Args:
        pull (dict): PR dictionary object from GitHub
        repo (Repo): Repo object
    Return:
        text (str): problem statement
        hints (str): hints
    """
    text = ""
    all_hints_text = list()
    for issue_number in pull["resolved_issues"]:
        url = f"https://code.djangoproject.com/ticket/{issue_number}"
        resp = requests.get(url)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get problem statement (title + body)
        issue_desc = soup.find("div", {"id": "ticket"})
        title = issue_desc.find("h1", class_="searchable").get_text()
        title = re.sub(r"\s+", " ", title).strip()
        body = issue_desc.find("div", class_="description").get_text()
        body = re.sub(r"\n+", "\n", body)
        body = re.sub(r"    ", "\t", body)
        body = re.sub(r"[ ]{2,}", " ", body).strip()
        text += f"{title}\n{body}\n"

        # Get time of first commit in PR
        commits = repo.get_all_loop(
            repo.api.pulls.list_commits, pull_number=pull["number"], quiet=True
        )
        commits = list(commits)
        if len(commits) == 0:
            continue
        commit_time = commits[0].commit.author.date
        commit_time = time.mktime(time.strptime(commit_time, "%Y-%m-%dT%H:%M:%SZ"))

        # Get all comments before first commit
        comments_html = soup.find("div", {"id": "changelog"})
        div_blocks = comments_html.find_all("div", class_="change")
        # Loop through each div block
        for div_block in div_blocks:
            # Find the comment text and timestamp
            comment_resp = div_block.find("div", class_="comment")
            timestamp_resp = div_block.find("a", class_="timeline")
            if comment_resp is None or timestamp_resp is None:
                continue

            comment_text = re.sub(r"\s+", " ", comment_resp.text).strip()
            timestamp = timestamp_resp["title"]
            if timestamp.startswith("See timeline at "):
                timestamp = timestamp[len("See timeline at ") :]
            if "/" in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%m/%d/%y %H:%M:%S"))
            elif "," in timestamp:
                timestamp = time.mktime(time.strptime(timestamp, "%b %d, %Y, %I:%M:%S %p"))
            else:
                raise ValueError(f"Timestamp format not recognized: {timestamp}")

            # Append the comment and timestamp as a tuple to the comments list
            if timestamp < commit_time:
                all_hints_text.append((comment_text, timestamp))

    return text, all_hints_text



def create_instance(repo: Repo, pull: dict) -> dict:
    """
    Create a single task instance from a pull request, where task instance is:

    {
        repo (str): owner/repo this task instance is from,
        pull_number (int): number of PR this task instance is from,
        base_commit (str): SHA of the base commit PR is based on,
        patch (str): reference solution as .patch (apply to base commit),
        test_patch (str): test suite as .patch (apply to base commit),
    }
    """
    patch, test_patch = extract_patches(pull, repo)
    problem_info = extract_problem_info(pull, repo)

    # problem_statement, hints = extract_problem_statement_and_hints(pull, repo)
    return {
        "repo": repo.repo.full_name,
        "pull_number": pull["number"],
        "url": pull["html_url"],
        "instance_id": (repo.repo.full_name + "-" + str(pull["number"])).replace(
            "/", "__"
        ),
        "issue_numbers": pull["resolved_issues"],
        "base_commit": pull["base"]["sha"],
        "patch": patch,
        "test_patch": test_patch,
        "problem_info": problem_info,
        "problem_statement": "[BLANK]",
        "hints_text": "[BLANK]",
        "created_at": pull["created_at"],
    }


import json
import os
import re
import requests
import subprocess

from datetime import datetime
from dotenv import load_dotenv
import git

# Constants - Task Instance Requirements File Paths
MAP_REPO_TO_REQS_PATHS = {
    "django/django": ["tests/requirements/py3.txt"],
    "matplotlib/matplotlib": ["requirements/dev/dev-requirements.txt", "requirements/testing/travis_all.txt"],
    "pallets/flask": ["requirements/dev.txt"],
    "pylint-dev/pylint": ["requirements_test.txt"],
    "pyvista/pyvista": ["requirements_test.txt", 'requirements.txt'],
    "sqlfluff/sqlfluff": ["requirements_dev.txt"],
    "sympy/sympy": ["requirements-dev.txt"],
}

# Constants - Task Instance environment.yml File Paths
MAP_REPO_TO_ENV_YML_PATHS = {
    "matplotlib/matplotlib": ["environment.yml"],
    "pydata/xarray": ["ci/requirements/environment.yml", "environment.yml"],
}
NON_TEST_EXTS = [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml"]
SWE_BENCH_URL_RAW = "https://raw.githubusercontent.com/"

load_dotenv()


def get_conda_env_names(conda_source: str, env: dict = None) -> list:
    """
    Get list of conda environment names for given conda path

    Args:
        conda_source (str): Path to conda executable
    Returns:
        env_names (list): List of conda environment names
    """
    # Get list of conda environments
    try:
        conda_envs = subprocess.run(
            f"{conda_source} env list".split(" "), check=True, capture_output=True, text=True, env=env,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Error stdout: {e.stdout}")
        print(f"Error stderr: {e.stderr}")
        raise e
    output = conda_envs.stdout
    lines = output.split("\n")
    # Store environment names to list
    env_names = []
    for line in lines:
        if line.startswith("#"):
            continue
        if line.strip() == "":
            continue
        parts = line.split()
        if len(parts) <= 1:
            continue
        env_name = parts[1]
        env_names.append(env_name)
    return env_names


def get_environment_yml(
        instance: dict,
        env_name: str,
        save_path: str = None,
        python_version: str = None,
    ) -> str:
    """
    Get environment.yml for given task instance

    Args:
        instance (dict): SWE Bench Task instance
        env_name (str): Rename retrieved environment.yml to this name
        save_path (str): If provided, save environment.yml to this path
    Returns:
        environment.yml (str): If save_path given, returns path to saved environment.yml.
            Otherwise, returns environment.yml as string
    """
    # Attempt to find environment.yml at each path based on task instance's repo
    path_worked = False

    commit = 'environment_setup_commit' if 'environment_setup_commit' in instance else 'base_commit'
    for req_path in MAP_REPO_TO_ENV_YML_PATHS[instance["repo"]]:
        reqs_url = os.path.join(
            SWE_BENCH_URL_RAW, instance["repo"], instance[commit], req_path
        )
        reqs = requests.get(reqs_url)
        if reqs.status_code == 200:
            path_worked = True
            break
    if not path_worked:
        print(
            f"Could not find environment.yml at paths {MAP_REPO_TO_ENV_YML_PATHS[instance['repo']]}"
        )
        return None

    lines = reqs.text.split("\n")
    cleaned = []
    for line in lines:
        # Rename environment to given name
        if line.startswith("name:"):
            cleaned.append(f"name: {env_name}")
            continue
        if line.startswith("dependencies:"):
            cleaned.append(line)
            if python_version is not None:
                cleaned.append(f"  - python={python_version}")
            continue
        cleaned.append(line)

    # Return environment.yml as string if no save path given
    if save_path is None:
        return "\n".join(cleaned)

    # Save environment.yml to given path and return path
    path_to_reqs = os.path.join(save_path, "environment.yml")
    with open(path_to_reqs, "w") as f:
        f.write("\n".join(cleaned))
    return path_to_reqs


def get_instances(instance_path: str) -> list:
    """
    Get task instances from given path

    Args:
        instance_path (str): Path to task instances
    Returns:
        task_instances (list): List of task instances
    """
    if any([instance_path.endswith(x) for x in [".jsonl", ".jsonl.all"]]):
        task_instances = list()
        with open(instance_path) as f:
            for line in f.readlines():
                task_instances.append(json.loads(line))
        return task_instances

    with open(instance_path) as f:
        task_instances = json.load(f)
    return task_instances


def get_requirements(instance: dict, save_path: str = None):
    """
    Get requirements.txt for given task instance

    Args:
        instance (dict): task instance
        save_path (str): If provided, save requirements.txt to this path
    Returns:
        requirements.txt (str): If save_path given, returns path to saved requirements.txt.
            Otherwise, returns requirements.txt as string
    """
    # Attempt to find requirements.txt at each path based on task instance's repo
    path_worked = False
    commit = 'environment_setup_commit' if 'environment_setup_commit' in instance else 'base_commit'

    for req_path in MAP_REPO_TO_REQS_PATHS[instance["repo"]]:
        reqs_url = os.path.join(
            SWE_BENCH_URL_RAW, instance["repo"], instance[commit], req_path
        )
        reqs = requests.get(reqs_url)
        if reqs.status_code == 200:
            path_worked = True
            break
    if not path_worked:
        print(
            f"Could not find requirements.txt at paths {MAP_REPO_TO_REQS_PATHS[instance['repo']]}"
        )
        return None

    lines = reqs.text
    original_req = []
    additional_reqs = []
    req_dir = "/".join(req_path.split("/")[:-1])
    exclude_line = lambda line: any(
        [line.strip().startswith(x) for x in ["-e .", "#", ".[test"]]
    )

    for line in lines.split("\n"):
        if line.strip().startswith("-r"):
            # Handle recursive requirements
            file_name = line[len("-r") :].strip()
            reqs_url = os.path.join(
                SWE_BENCH_URL_RAW,
                instance["repo"],
                instance[commit],
                req_dir,
                file_name,
            )
            reqs = requests.get(reqs_url)
            if reqs.status_code == 200:
                for line_extra in reqs.text.split("\n"):
                    if not exclude_line(line_extra):
                        additional_reqs.append(line_extra)
        else:
            if not exclude_line(line):
                original_req.append(line)

    # Combine all requirements into single text body
    additional_reqs.append("\n".join(original_req))
    all_reqs = "\n".join(additional_reqs)

    # print(all_reqs)

    replacements = {
        # See https://github.com/princeton-nlp/SWE-bench/issues/199
        # This package was sinced yanked, so we need to force pip
        # to install it.
        "types-pkg_resources": "types-pkg-resources==0.1.3",
    }
    requirements = [req.strip() for req in all_reqs.split("\n") if req.strip()]
    requirements_replaced = []
    for requirement in requirements:
        if requirement in replacements:
            print(f"Replaced {requirement!r} with {replacements[requirement]!r} (replace_uninstallable_packages)")
            requirements_replaced.append(replacements[requirement])
        else:
            requirements_replaced.append(requirement)
    all_reqs = "\n".join(requirements_replaced) + "\n"

    # print(all_reqs)

    if save_path is None:
        return all_reqs

    path_to_reqs = os.path.join(save_path, "requirements.txt")
    with open(path_to_reqs, "w") as f:
        f.write(all_reqs)
    return path_to_reqs


def get_test_directives(instance: dict) -> list:
    """
    Get test directives from the test_patch of a task instance

    Args:
        instance (dict): task instance
    Returns:
        directives (list): List of test directives
    """
    # HumanEvalFix: For seq2seq code repos, testing command is fixed
    if any([
        x == instance["repo"] for x in
        ["swe-bench/humaneval", "swe-bench/humanevalfix-python"]
    ]):
        return ["test.py"]
    if any([
        x == instance["repo"] for x in
        ["swe-bench/humanevalfix-go", "swe-bench/humanevalfix-java"]
    ]):
        return []
    if instance["repo"] == "swe-bench/humanevalfix-js":
        return ["test.js"]

    # Get test directives from test patch and remove non-test files
    diff_pat = r"diff --git a/.* b/(.*)"
    test_patch = instance["test_patch"]
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]

    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots (module referencing)
    if instance["repo"] == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def clone_repo(repo_name: str, path: str, token: str = None) -> bool:
    """
    Wrapper for cloning repo from swe-bench organization

    Args:
        repo_name (str): Name of repo to clone
        path (str): Path to clone repo to
        token (str): GitHub token to use for cloning
    Returns:
        success (bool): True if repo cloned successfully, False otherwise
    """
    try:
        if token is None:
            token = os.environ.get("GITHUB_TOKEN", "git")
        # repo_url = (
        #     f"https://{token}@github.com/swe-bench/"
        #     + repo_name.replace("/", "__")
        #     + ".git"
        # )
        repo_url = f"https://github.com/{repo_name}.git"
        git.Repo.clone_from(repo_url, path)
        return True
    except Exception as e:
        print(e)
        return False


def split_instances(input_list: list, n: int) -> list:
    """
    Split a list into n approximately equal length sublists

    Args:
        input_list (list): List to split
        n (int): Number of sublists to split into
    Returns:
        result (list): List of sublists
    """
    avg_length = len(input_list) // n
    remainder = len(input_list) % n
    result, start = [], 0

    for i in range(n):
        length = avg_length + 1 if i < remainder else avg_length
        sublist = input_list[start : start + length]
        result.append(sublist)
        start += length

    return result


def find_python_by_date(target_date, date_format="%Y%m%d"):
    """
    Find python version closest to given date

    Args:
        target_date (str): Date to find python version for
        date_format (str): Format of target_date
    Returns:
        python_version (str): Python version closest to target_date
    """
    # Make web request to versions + date page
    url = f"https://www.python.org/doc/versions/"
    response = requests.get(url)

    # Look for all matches
    pattern = r"Python (.*)</a>, documentation released on (.*)\.</"
    matches = re.findall(pattern, response.text)

    # Convert NL dates to date time format
    def convert_to_yyyymmdd(input_date):
        # Parse the input date string using datetime
        date_obj = datetime.strptime(input_date, date_format)
        # Format the date object into YYYYMMDD format
        return date_obj.strftime("%Y%m%d")

    version_to_date = [(match[0], convert_to_yyyymmdd(match[1])) for match in matches]

    # Find Python
    for x in version_to_date:
        if target_date >= x[1]:
            return x[0]
    return None


class DotDict:
    """
    Wrapper class for accessing dictionary keys as attributes
    """

    def __init__(self, data):
        self.data = data

    def __getattr__(self, key):
        return self.data.get(key)


### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta


def extract_minimal_patch(model_patch):
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def has_attribute_or_import_error(log_before):
    """
    Check to see if Attribute/Import-prefix is in log text

    Args:
        log_before (str): Validation log text before patch application
    """
    log_before = log_before.lower()

    if any([x in log_before for x in ['attribute', 'import']]):
        def get_lines_with_word(text, target_word):
            # Function to extract line(s) that contains target_word
            text, target_word = text.lower(), target_word.lower()
            lines, hits = text.split('\n')[::-1], []
            for line in lines:
                if target_word in line:
                    hits.append(line)
            return hits
        
        # Get line with Attribute/Import error
        lines_1 = get_lines_with_word(log_before, 'attribute')
        lines_2 = get_lines_with_word(log_before, 'import')
        lines_1 = " ".join(lines_1)
        lines_2 = " ".join(lines_2)

        if any([(x in lines_1 or x in lines_2) for x in ['error', 'fail']]):
            return True
    return False
