from dria_agent.agent.tool import tool
from typing import List, Dict

try:
    from github import Github
except ImportError:
    raise ImportError("Please run pip install 'dria_agent[tools]'")

import os


def get_client():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise Exception("Please set GITHUB_TOKEN environment variable")
    return Github(token)


@tool
def list_repositories(user: str = None, org: str = None) -> List[Dict]:
    """
    List repositories for a user or organization.

    :param user: GitHub username.
    :param org: GitHub organization name.
    :return: List of dicts with repo's 'name', 'full_name', and 'private' status.
    """
    client = get_client()
    if org:
        repos = client.get_organization(org).get_repos()
    elif user:
        repos = client.get_user(user).get_repos()
    else:
        repos = client.get_user().get_repos()
    return [
        {"name": r.name, "full_name": r.full_name, "private": r.private} for r in repos
    ]


@tool
def create_repository(name: str, description: str = "", private: bool = False) -> dict:
    """
    Create a new repository for the authenticated user.

    :param name: Repository name.
    :param description: Repository description.
    :param private: Whether the repo is private.
    :return: Repository details.
    """
    client = get_client()
    user = client.get_user()
    repo = user.create_repo(name, description=description, private=private)
    return {"name": repo.name, "full_name": repo.full_name, "private": repo.private}


@tool
def delete_repository(full_name: str) -> bool:
    """
    Delete a repository.

    :param full_name: Full repository name (e.g., "user/repo").
    :return: True if deletion was successful.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    repo.delete()
    return True


@tool
def get_repository_info(full_name: str) -> dict:
    """
    Get detailed information about a repository.

    :param full_name: Full repository name.
    :return: Repo details including description, stars, forks, and open issues.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    return {
        "name": repo.name,
        "full_name": repo.full_name,
        "description": repo.description,
        "private": repo.private,
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "open_issues": repo.open_issues_count,
    }


@tool
def list_issues(full_name: str, state: str = "open") -> List[Dict]:
    """
    List issues for a repository.

    :param full_name: Full repository name.
    :param state: Issue state ('open', 'closed', or 'all').
    :return: List of issues (dict) with number, title, and state.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    issues = repo.get_issues(state=state)
    return [{"number": i.number, "title": i.title, "state": i.state} for i in issues]


@tool
def create_issue(full_name: str, title: str, body: str = None) -> dict:
    """
    Create an issue in a repository.

    :param full_name: Full repository name.
    :param title: Issue title.
    :param body: Issue body.
    :return: Created issue details.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    issue = repo.create_issue(title=title, body=body)
    return {"number": issue.number, "title": issue.title, "state": issue.state}


@tool
def close_issue(full_name: str, issue_number: int) -> bool:
    """
    Close an issue in a repository.

    :param full_name: Full repository name.
    :param issue_number: Issue number.
    :return: True if the issue was closed.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    issue = repo.get_issue(number=issue_number)
    issue.edit(state="closed")
    return True


@tool
def merge_pull_request(
    full_name: str, pr_number: int, commit_message: str = None
) -> dict:
    """
    Merge a pull request in a repository.

    :param full_name: Full repository name.
    :param pr_number: Pull request number.
    :param commit_message: Optional commit message.
    :return: Merge result details.
    """
    client = get_client()
    repo = client.get_repo(full_name)
    pr = repo.get_pull(pr_number)
    merge_result = pr.merge(commit_message=commit_message)
    return {"merged": merge_result.merged, "message": merge_result.message}


GITHUB_TOOLS = [
    list_repositories,
    create_repository,
    delete_repository,
    get_repository_info,
    list_issues,
    create_issue,
    close_issue,
    merge_pull_request,
]
