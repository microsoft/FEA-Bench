import json
import logging
import os
import traceback
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unidiff
from tqdm.auto import tqdm
import re
import ast
import shutil
import chardet
import subprocess
from argparse import ArgumentTypeError
from git import Repo
from pathlib import Path
from tempfile import TemporaryDirectory


import tiktoken

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)



def cl100k(text, tokenizer):
    return tokenizer.encode(text, disallowed_special=())

TOKENIZER_FUNCS = {
    "cl100k": (tiktoken.get_encoding("cl100k_base"), cl100k),
}


OLD_PATCH_EXAMPLE = """--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points"""


OLD_FULL_GENERATION_EXAMPLE = """[start of /src/this_file.py]
import os

def euclidean(a, b):
    if b == 0:
        return a
    return euclidean(b, a % b)
[end of /src/this_file.py]
[start of /src/another_file.py]
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points
[end of /src/another_file.py]"""


WRAPPED_EDIT_GENERATION_EXAMPLE = """```
<edit>
[start of the snippet before editing in src/code.py]
def factorial(a):
    res = 1
    while a >= 0:
        res *= a
    return res
[end of the snippet before editing in src/code.py]

[start of the snippet after editing in src/code.py]
def factorial(a):
    assert type(a) == int and a >= 0
    res = 1
    while a >= 2:
        res *= a
        a -= 1
    return res
[end of the snippet after editing in src/code.py]
</edit>

<edit>
[start of the snippet before editing in src/code.py]
def exact_dividion(x, y):
    return x % y == 0
[end of the snippet before editing in src/code.py]

[start of the snippet after editing in src/code.py]
def exact_dividion(x, y):
    assert type(x) == type(y) == int and x > 0 and y > 0
    return x % y == 0
[end of the snippet after editing in src/code.py]
</edit>

<edit>
[start of the snippet before editing in src/demo.py]
[end of the snippet before editing in src/demo.py]

[start of the snippet after editing in src/demo.py]
from code import factorial
print(factorial(5))
[end of the snippet after editing in src/demo.py]
</edit>
```"""


PATCH_GENERATION_EXAMPLE = """diff --git a/src/code.py b/src/code.py
--- a/src/code.py
+++ b/src/code.py
@@ -1,11 +1,14 @@
 def factorial(a):
+    assert type(a) == int and a >= 0
     res = 1
-    while a >= 0:
+    while a >= 2:
         res *= a
+        a -= 1
     return res
     #
     # 
     # 
     # 
 def exact_dividion(x, y):
+    assert type(x) == type(y) == int and x > 0 and y > 0
     return x % y == 0
diff --git a/src/demo.py b/src/demo.py
new file mode 100644
--- /dev/null
+++ b/src/demo.py
@@ -0,0 +1,2 @@
+from code import factorial
+print(factorial(5))"""


def add_lines_list(content):
    content_with_lines = list()
    for ix, line in enumerate(content.split("\n"), start=1):
        content_with_lines.append(f"{ix} {line}")
    return content_with_lines


def add_lines(content):
    return "\n".join(add_lines_list(content))


def make_code_text(files_dict, add_line_numbers=True):
    all_text = ""
    for filename, contents in sorted(files_dict.items()):
        if is_test_file(filename) or (str(filename).endswith((".rst", ".md")) and contents == None):
            continue
        all_text += f"[start of {filename}]\n"
        if contents:     # for empty file with None value, only reserve the markers of start and the end
            if add_line_numbers:
                all_text += add_lines(contents)
            else:
                all_text += contents
            all_text += f"\n[end of {filename}]\n\n"
        else:
            all_text += f"[end of {filename}]\n\n"

    return all_text


def make_code_text_edits_only(files_dict, patch, add_line_numbers=True):
    files = dict()
    patch = unidiff.PatchSet(patch)
    for patched_file in patch:
        source_file = patched_file.source_file.split("a/", 1)[-1]
        files[source_file] = list()
        for hunk in patched_file:
            start = hunk.source_start - 15
            end = start + hunk.source_length + 15
            files[source_file].append((start, end))
    all_text = ""
    for filename, content in files_dict.items():
        all_text += f"[start of {filename}]\n"
        content_with_lines = add_lines_list(content)
        for start, end in files[filename]:
            if start > 0:
                all_text += "...\n"
            all_text += "\n".join(content_with_lines[start:end])
            all_text += "\n"
            if end < len(content_with_lines):
                all_text += "...\n"
        all_text = all_text.strip("\n")
        all_text += f"\n[end of {filename}]\n"
    return all_text.strip("\n")


def get_wrapped_doc_changes(doc_patch):
    if doc_patch.strip():
        readme_content = "To implement the new features mentioned above, some design in this repository need to be modified, as the modification in document files:\n<description changes>\n" + doc_patch.strip() + "\n</description changes>"
    else:
        readme_content = ""
    return readme_content


def get_wrapped_request_content(task):
    # pr content
    if "pull_request_text" in task:
        return task["pull_request_text"]
    
    first_commit_time = task["problem_info"]["first_commit_time"]
    request_content = str(task["problem_info"]["pr_title"]) + "\n" + str(task["problem_info"]["pr_body"]) + "\n----------"
    for pr_comment in task["problem_info"]["pr_timeline"]:
        if pr_comment["time"] < first_commit_time:
            request_content += ('\n' + pr_comment["comment"])
    request_content = "<request>\n" + request_content.strip() + "\n</request>"

    return request_content


def get_wrapped_discussion_content(task):
    # discussion content
    if "issue_text" in task:
        return task["issue_text"]

    first_commit_time = task["problem_info"]["first_commit_time"]
    discussion = ""
    for issue_number in task["problem_info"]["issues"]:
        if not task["problem_info"]["issues"][issue_number]:
            continue
        discussion += f'{task["problem_info"]["issues"][issue_number]["issue_title"].strip()}\n{task["problem_info"]["issues"][issue_number]["issue_body"].strip()}\n----------'
        for issue_comment in task["problem_info"]["issues"][issue_number]["issue_timeline"]:
            if issue_comment["time"] < first_commit_time:
                discussion += ('\n' + issue_comment["comment"].strip())
        discussion += "\n--------------------"
    if discussion:
        discussion = "Here is the discussion in the issues of the pull request.\n" + "<issues>\n" + discussion + "\n</issues>"
    return discussion


def get_wrapped_definitions(task, with_doc=True):
    # added definitions
    definitions = "There are several new functions or classes that need to be implemented, using the definitions below: \n<definitions>\n{def_list}\n</definitions>"
    temp_definitions = ""
    added_components = task["new_components"]
    for file in added_components:
        temp_definitions += f"[start of new definitions in {file}]\n"
        for item in added_components[file]:
            temp_definitions += f"(definition of {item['name']}:)\n"
            temp_definitions += f"{item['signature']}\n"
            if item['doc'] and with_doc:
                temp_definitions += f'''"""{item['doc']}"""\n'''
            # temp_definitions += f"\n"
        temp_definitions += f"[end of new definitions in {file}]\n"

    definitions = definitions.format(def_list=temp_definitions)
    return definitions


NATURAL_FORMAT_INSTRUCTION = '''
Please solve the feature request with adding the functions or classes between the <definitions> and </definitions>. You do not need to output any changes or edits for description files like .md or .rst files.
I need you to make multiple edits across one or more files in a repository to implement a specific feature or improvement mentioned above. 
For each edit, output the changes in the following format:

```
<edit>
[start of the snippet before editing in <file_path>]
<code_before_edit>
[end of the snippet before editing in <file_path>]

[start of the snippet after editing in <file_path>]
<code_after_edit>
[end of the snippet after editing in <file_path>]
</edit>
```

Notes:
- The <file_path> is the relative path of the file being edited.
- The <code_before_edit> snippet should include several lines before and after the modified region, unless the file was originally empty.
- If a file was originally empty, leave <code_before_edit> blank but ensure <code_after_edit> includes the new content.
- Ensure the edits are sequential and address all necessary changes to achieve the requested feature.

Here is an example of the output format:
'''.strip()


PATCH_FORMAT_INSTRUCTION = '''
Please solve the feature request with adding the functions or classes between the <definitions> and </definitions>. You do not need to output any changes or edits for description files like .md or .rst files.
I need you to make multiple edits across one or more files in a repository to implement a specific feature or improvement mentioned above. 
The edits should be output in patch format.
'''.strip()


def prompt_natural_edit_full(instance):
    premise = "You will be provided with a partial code base and an feature request which requires a new feature to add in the code repository."

    request_text = get_wrapped_request_content(instance)
    issue_text = get_wrapped_discussion_content(instance)

    readme_changes = get_wrapped_doc_changes(instance["non_py_patch"])
    definitions_text = get_wrapped_definitions(instance)

    readmes_text = make_code_text(instance["readmes"], add_line_numbers=False)
    code_text = make_code_text(instance["files"], add_line_numbers=False)

    file_hint = "\nAll the involved readme files and code files are listed below with their contents. If a file's content is indicated as empty, it means that the file does not yet exist in the current repository and needs to be created with new content.\n"

    final_instruction = "I need you to solve the feature request with a series of edits in the format shown above."

    final_text = [
        premise,
        request_text,
        issue_text,
        "",
        readme_changes,
        "",
        definitions_text,
        "",
        file_hint,
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        NATURAL_FORMAT_INSTRUCTION,
        "",
        WRAPPED_EDIT_GENERATION_EXAMPLE,
        "",
        final_instruction,
        "Respond below:",
    ]
    final_text = "\n".join(final_text)
    return final_text


def prompt_diff_full(instance):
    premise = "You will be provided with a partial code base and an feature request which requires a new feature to add in the code repository."

    request_text = get_wrapped_request_content(instance)
    issue_text = get_wrapped_discussion_content(instance)

    readme_changes = get_wrapped_doc_changes(instance["non_py_patch"])
    definitions_text = get_wrapped_definitions(instance)

    readmes_text = make_code_text(instance["readmes"], add_line_numbers=True)
    code_text = make_code_text(instance["files"], add_line_numbers=True)


    file_hint = "\nAll the involved readme files and code files are listed below with their contents. If a file's content is indicated as empty, it means that the file does not yet exist in the current repository and needs to be created with new content.\n"

    example_explanation = (
        f"Here is an example of a patch file. It consists of changes to the code base. "
        + f"It specifies the file names, the line numbers of each change, and the removed and added lines. "
        + f"A single patch file can contain changes to multiple files."
    )
    final_instruction = (
        f"I need you to solve the provided feature request by generating a single patch file that I can apply "
        + f"directly to this repository using git apply. Please respond with a single patch "
        + f"file in the format shown above."
    )

    final_text = [
        premise,
        request_text,
        issue_text,
        "",
        readme_changes,
        "",
        definitions_text,
        "",
        file_hint,
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        PATCH_FORMAT_INSTRUCTION,
        example_explanation,
        "<patch>",
        PATCH_GENERATION_EXAMPLE,
        "</patch>",
        "",
        final_instruction,
        "Respond below:",
    ]
    final_text = "\n".join(final_text)
    return final_text


def prompt_natural_edit_wo_doc(instance):
    premise = "You will be provided with a partial code base and an feature request which requires a new feature to add in the code repository."

    request_text = get_wrapped_request_content(instance)
    issue_text = get_wrapped_discussion_content(instance)

    readme_changes = get_wrapped_doc_changes(instance["non_py_patch"])
    definitions_text = get_wrapped_definitions(instance, with_doc=False)

    readmes_text = make_code_text(instance["readmes"], add_line_numbers=False)
    code_text = make_code_text(instance["files"], add_line_numbers=False)

    file_hint = "\nAll the involved readme files and code files are listed below with their contents. If a file's content is indicated as empty, it means that the file does not yet exist in the current repository and needs to be created with new content.\n"

    final_instruction = "I need you to solve the feature request with a series of edits in the format shown above."

    final_text = [
        premise,
        request_text,
        issue_text,
        "",
        # readme_changes,
        definitions_text,
        "",
        file_hint,
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        NATURAL_FORMAT_INSTRUCTION,
        "",
        WRAPPED_EDIT_GENERATION_EXAMPLE,
        "",
        final_instruction,
        "Respond below:",
    ]
    final_text = "\n".join(final_text)
    return final_text


def prompt_diff_wo_doc(instance):
    premise = "You will be provided with a partial code base and an feature request which requires a new feature to add in the code repository."

    request_text = get_wrapped_request_content(instance)
    issue_text = get_wrapped_discussion_content(instance)

    readme_changes = get_wrapped_doc_changes(instance["non_py_patch"])
    definitions_text = get_wrapped_definitions(instance, with_doc=False)

    readmes_text = make_code_text(instance["readmes"], add_line_numbers=True)
    code_text = make_code_text(instance["files"], add_line_numbers=True)


    file_hint = "\nAll the involved readme files and code files are listed below with their contents. If a file's content is indicated as empty, it means that the file does not yet exist in the current repository and needs to be created with new content.\n"

    example_explanation = (
        f"Here is an example of a patch file. It consists of changes to the code base. "
        + f"It specifies the file names, the line numbers of each change, and the removed and added lines. "
        + f"A single patch file can contain changes to multiple files."
    )
    final_instruction = (
        f"I need you to solve the provided feature request by generating a single patch file that I can apply "
        + f"directly to this repository using git apply. Please respond with a single patch "
        + f"file in the format shown above."
    )

    final_text = [
        premise,
        request_text,
        issue_text,
        "",
        # readme_changes,
        definitions_text,
        "",
        file_hint,
        "<code>",
        readmes_text,
        code_text,
        "</code>",
        "",
        PATCH_FORMAT_INSTRUCTION,
        example_explanation,
        "<patch>",
        PATCH_GENERATION_EXAMPLE,
        "</patch>",
        "",
        final_instruction,
        "Respond below:",
    ]
    final_text = "\n".join(final_text)
    return final_text


def ingest_files(filenames):
    files = dict()
    for filename in filenames:
        with open(filename) as f:
            content = f.read()
        files[filename] = content
    return files


PROMPT_FUNCTIONS = {
    # "style-2": prompt_style_2,
    # "style-3": prompt_style_3,
    # "full_file_gen": full_file_gen,
    # "style-2-edits-only": prompt_style_2_edits_only,
    "diff": prompt_diff_full,
    "natural_edit": prompt_natural_edit_full,
    "diff_wo_doc": prompt_diff_wo_doc,
    "natural_edit_wo_doc": prompt_natural_edit_wo_doc,
}


def add_retrieval_results(input_instances, retrieval_file, k, file_source):
    """
    Adds retrieval results to input_instances in-place
    """
    retrieval_results_path = Path(retrieval_file)
    assert (
        retrieval_results_path.exists()
    ), f"Retrieval results not found at {retrieval_results_path}"
    retrieval_results = [json.loads(line) for line in open(retrieval_results_path)]
    retrieval_results = {x["instance_id"]: x["hits"] for x in retrieval_results}
    for instance_id, instance in tqdm(
        input_instances.items(),
        total=len(input_instances),
        desc="Adding retrieval results",
    ):
        try:
            instance["hits"] = retrieval_results[instance_id][:k]
        except KeyError:
            logger.warning(f"Instance {instance_id} not found in retrieval results")
            instance["hits"] = list()


def get_oracle_filenames(instance):
    """
    Returns the filenames that are changed in the patch
    """
    source_files = {
        patch_file.source_file.split("a/", 1)[-1]
        for patch_file in unidiff.PatchSet(instance["patch"])
    }
    gold_docs = set()
    for source_file in source_files:
        gold_docs.add(source_file)
    return gold_docs


def add_text_inputs(
    input_instances,
    retrieval_file,
    k,
    prompt_style,
    file_source,
    max_context_len=None,
    tokenizer_name=None,
    verbose=False,
):
    """Adds text inputs context for prediction in-place.

    Args:
    - input_instances: dictionary with unprocessed input instances.
    - retrieval_file: if using retrieval method for file_contents, specify retrieval_file to add retrieval results
    - k: if using retrieval, specifies the maximum number of files to included within context
    - prompt_style: specify the function to generate instructions and prompt provided an instance (from PROMPT_FUNCTIONS)
    - file_source: where to collect file_contents (e.g. oracle or bm25)
    - verbose: set ContextManager verbose to True
    """
    if max_context_len is not None:
        assert (
            tokenizer_name is not None
        ), "Must specify tokenizer_name if using max_context_len"
        tokenizer, tokenizer_func = TOKENIZER_FUNCS[tokenizer_name]
    input_instances_copy = deepcopy(input_instances)
    if file_source in {"bm25"}:
        add_retrieval_results(input_instances_copy, retrieval_file, k, file_source)
    orig_dir = os.getcwd()
    with TemporaryDirectory(
        dir="/scratch" if os.path.exists("/scratch") else "/tmp"
    ) as root_dir:
        for instance_id, instance in tqdm(
            input_instances_copy.items(),
            total=len(input_instances_copy),
            desc="Adding text inputs",
        ):
            try:
                with AutoContextManager(
                    instance, root_dir, verbose=verbose
                ) as cm:
                    readmes = cm.get_readme_files()
                    instance["readmes"] = ingest_files(readmes)
                    if max_context_len is not None:
                        instance["file_contents"] = dict()
                        base_text_inputs = PROMPT_FUNCTIONS[prompt_style](instance)
                        base_text_input_length = len(
                            tokenizer_func(base_text_inputs, tokenizer)
                        )
                    if file_source in {"oracle"}:
                        instance["file_contents"] = ingest_files(
                            get_oracle_filenames(instance)
                        )
                    elif file_source in {"bm25"}:
                        instance["file_contents"] = ingest_files(
                            [x["docid"] for x in instance["hits"]]
                        )
                    elif file_source in {"all"}:
                        instance["file_contents"] = ingest_directory_contents(
                            cm.repo_path
                        )
                    elif file_source in {"none"}:
                        instance["file_contents"] = dict()
                    else:
                        raise ValueError(f"Invalid file source {file_source}")
                    if max_context_len is not None:
                        cur_input_len = base_text_input_length
                        include_files = list()
                        for filename in [x["docid"] for x in instance["hits"]]:
                            content = make_code_text(
                                {filename: instance["file_contents"][filename]}
                            )
                            if tokenizer_name in {"llama"}:
                                tokens = tokenizer_func("\n" + content, tokenizer)
                                idx = tokens.index(13)
                                assert (
                                    idx <= 2
                                ), "Expected newline token id (13) to be one of the first three tokens"
                                tokens = tokens[idx + 1 :]  # remove newline tokens
                            else:
                                tokens = tokenizer_func(content, tokenizer)
                            if cur_input_len + len(tokens) < max_context_len:
                                include_files.append(filename)
                                cur_input_len += len(tokens)
                        instance["file_contents"] = {
                            filename: instance["file_contents"][filename]
                            for filename in include_files
                        }
                    input_instances[instance_id]["text_inputs"] = PROMPT_FUNCTIONS[
                        prompt_style
                    ](instance)
            except Exception as e:
                print(f"Failed on instance {instance_id}", e)
                traceback.print_exc()
                input_instances[instance_id]["text_inputs"] = None
            finally:
                # if AutoContextManager fails to exit properly future exits will return the wrong directory
                os.chdir(orig_dir)
    os.chdir(orig_dir)


COLUMN_TO_PROMPT = {
    "patch-detailed": "diff",
    "natural-detailed": "natural_edit",
    "patch-brief": "diff_wo_doc",
    "natural-brief": "natural_edit_wo_doc",
}


def add_text_inputs_bm25(
    input_instances,
    retrieval_file,
    k,
    max_context_len=None,
    tokenizer_name=None,
    verbose=False,
):
    
    """Adds text inputs context for prediction in-place.

    Args:
    - input_instances: dictionary with unprocessed input instances.
    - retrieval_file: if using retrieval method for file_contents, specify retrieval_file to add retrieval results
    - k: if using retrieval, specifies the maximum number of files to included within context
    - prompt_style: specify the function to generate instructions and prompt provided an instance (from PROMPT_FUNCTIONS)
    - file_source: where to collect file_contents (e.g. oracle or bm25)
    - verbose: set ContextManager verbose to True
    """
    if max_context_len is not None:
        assert (
            tokenizer_name is not None
        ), "Must specify tokenizer_name if using max_context_len"
        tokenizer, tokenizer_func = TOKENIZER_FUNCS[tokenizer_name]
    input_instances_copy = deepcopy(input_instances)
    
    add_retrieval_results(input_instances_copy, retrieval_file, k, 'bm25')
    orig_dir = os.getcwd()
    temp_dir = "/data/tmp"
    try:
        os.makedirs(temp_dir, exist_ok=True)
    except Exception as e:
        temp_dir = "/tmp"
    with TemporaryDirectory(
        dir=temp_dir
    ) as root_dir:
        for instance_id, instance in tqdm(
            input_instances_copy.items(),
            total=len(input_instances_copy),
            desc="Adding text inputs",
        ):
            try:
                with AutoContextManager(
                    instance, root_dir, verbose=verbose
                ) as cm:
                    readmes = cm.get_readme_files()
                    instance["readmes"] = ingest_files(readmes)

                    # files with added components must be placed into the context
                    assert instance["context_type"] == ORACLE, "instances loaded from dataset are not under oracle setting."
                    oracle_files_map = convert_files_list_into_map(instance["files"])
                    instance["new_components"] = convert_components_list_to_map(instance["new_components"])
                    instance["files"] = {}
                    essential_files = []
                    for file in instance["readmes"]:
                        essential_files.append(file)
                    
                    for file in instance["new_components"]:
                        essential_files.append(file)
                        instance["files"][file] = oracle_files_map[file]

                    # added files will also be put into context ?
                    for file in instance["added_files"]:
                        essential_files.append(file)
                        if oracle_files_map[file]: 
                            print(f"[Warning] Added file {file} content should be None value. But got {oracle_files_map[file]}")
                        instance["files"][file] = oracle_files_map[file]

                    if max_context_len is not None:
                        base_text_inputs = PROMPT_FUNCTIONS["natural_edit"](instance)
                        base_text_input_length = len(
                            tokenizer_func(base_text_inputs, tokenizer)
                        )

                    # add retrived files without repeating readme files or files with new components.
                    retrieved_files = [x["docid"] for x in instance["hits"] if x["docid"] not in essential_files]
                    retrieval_files_map = ingest_files(retrieved_files)

                    if max_context_len is not None:
                        cur_input_len = base_text_input_length
                        include_files = list()
                        for filename in retrieved_files:
                            content = make_code_text(
                                {filename: retrieval_files_map[filename]}
                            )

                            tokens = tokenizer_func(content, tokenizer)
                            if cur_input_len + len(tokens) < max_context_len:
                                include_files.append(filename)
                                cur_input_len += len(tokens)
                        # add to file map for all the included retrieval files
                        for filename in include_files:
                            instance["files"][filename] = retrieval_files_map[filename]
                    
                    for column in COLUMN_TO_PROMPT:
                        input_instances[instance_id][column] = str(PROMPT_FUNCTIONS[COLUMN_TO_PROMPT[column]](instance))

                    # # convert the new_components and files into the format of Sequence again to meet the requirements of Dataset
                    input_instances[instance_id]["context_type"] = "bm25"
                    input_instances[instance_id]["files"] = convert_files_map_into_list(instance["files"])
                    input_instances[instance_id]["readmes"] = convert_files_map_into_list(instance["readmes"])
                    # instance["new_components"] = convert_components_map_into_list(instance["new_components"])
                    

            except Exception as e:
                print(f"Failed on instance {instance_id}", e)
                traceback.print_exc()
                instance["context_type"] = None
                # input_instances[instance_id]["text"] = None
            finally:
                # if AutoContextManager fails to exit properly future exits will return the wrong directory
                os.chdir(orig_dir)
    os.chdir(orig_dir)
    

# INSTANCE TYPES
ORACLE = "oracle"

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


def checkout_to_commit(repo_dir, commit_sha):
    """Checkout the repository to a specific commit."""
    run_command(['git', 'checkout', '--quiet', commit_sha], cwd=repo_dir)


def apply_patch(repo_dir, patch_content):
    """Apply the patch content to the repository."""
    with open(os.path.join(repo_dir, 'patch.diff'), 'w') as f:
        f.write(patch_content)
    run_command(['git', 'apply', 'patch.diff'], cwd=repo_dir)
    os.remove(os.path.join(repo_dir, 'patch.diff'))


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


# @SWE-bench

NON_TEST_EXTS = [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml"]

def is_test_file(path):
    if any(
        test_word in path for test_word in
        ['test', 'tests', 'e2e', 'testing', 'check']
    ):
        if not any(path.endswith(ext) for ext in NON_TEST_EXTS):
            return True
    
    return False

def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\/|\.", name.lower()))
    return any(word in words for word in test_phrases)


def list_files(root_dir, include_tests=False):
    files = []
    for filename in Path(root_dir).rglob("*.py"):
        relative_filename = filename.relative_to(root_dir)
        if not include_tests and is_test(relative_filename.as_posix()):
            continue
        files.append(relative_filename.as_posix())
    return files


DIFF_PATTERN = re.compile(r"^diff(?:.*)")
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
    first_min = charlist.index('-')  if '-' in charlist else len(charlist)
    first_plus = charlist.index('+') if '+' in charlist else len(charlist)
    return min(first_min, first_plus)

def get_last_idx(charlist):
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1

def strip_content(hunk):
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split('\n')))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split('\n')[first_idx:last_idx]))
    new_hunk = '\n' + '\n'.join(new_lines) + '\n'
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
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


def repair_patch(model_patch):
    if model_patch is None:
        return None
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        diff_header = DIFF_PATTERN.findall(patch)
        if diff_header:
            new_patch += diff_header[0] + "\n"
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                *list(map(lambda x: int(x) if x.isnumeric() else x, hunk)), total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def extract_minimal_patch(model_patch):
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        diff_header = DIFF_PATTERN.findall(patch)
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(map(lambda x: int(x) if x.isnumeric() else x, hunk))
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return None
    diff_matches = []
    other_matches = []
    pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    return response.split("</s>")[0]


class ContextManager:
    def __init__(self, repo_path, base_commit, verbose=False):
        self.repo_path = Path(repo_path).resolve().as_posix()
        self.old_dir = os.getcwd()
        self.base_commit = base_commit
        self.verbose = verbose

    def __enter__(self):
        os.chdir(self.repo_path)
        cmd = f"git reset --hard {self.base_commit} && git clean -fdxq"
        if self.verbose:
            subprocess.run(cmd, shell=True, check=True)
        else:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return self

    def get_environment(self):
        raise NotImplementedError()  # TODO: activate conda environment and return the environment file

    def get_readme_files(self):
        files = os.listdir(self.repo_path)
        files = list(filter(lambda x: os.path.isfile(x), files))
        files = list(filter(lambda x: x.lower().startswith("readme"), files))
        return files

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.old_dir)


class AutoContextManager(ContextManager):
    """Automatically clones the repo if it doesn't exist"""

    def __init__(self, instance, root_dir=None, verbose=False, token=None):
        if token is None:
            token = os.environ.get("GITHUB_TOKEN", "git")
        self.tempdir = None
        if root_dir is None:
            self.tempdir = TemporaryDirectory()
            root_dir = self.tempdir.name
        self.root_dir = root_dir
        repo_dir = os.path.join(self.root_dir, instance["repo"].replace("/", "__"))
        if not os.path.exists(repo_dir):
            repo_url = (
                f"https://{token}@github.com/swe-bench/"
                + instance["repo"].replace("/", "__")
                + ".git"
            )
            repo_url = f"https://github.com/{instance['repo']}.git"
            if verbose:
                print(f"Cloning {instance['repo']} to {root_dir}")
            Repo.clone_from(repo_url, repo_dir)
        super().__init__(repo_dir, instance["base_commit"], verbose=verbose)
        self.instance = instance

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.tempdir is not None:
            self.tempdir.cleanup()
        return super().__exit__(exc_type, exc_val, exc_tb)



def get_imported_modules(filename):
    with open(filename) as file:
        tree = ast.parse(file.read(), filename)
    return [
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]


def resolve_module_to_file(module, level, root_dir):
    components = module.split(".")
    if level > 0:
        components = components[:-level]
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if dirpath.endswith(os.sep.join(components)):
            return [
                os.path.join(dirpath, filename)
                for filename in filenames
                if filename.endswith(".py")
            ]
    return []


def ingest_file_directory_contents(target_file, root_dir):
    imported_files = []
    files_to_check = [target_file]
    while files_to_check:
        current_file = files_to_check.pop()
        imported_files.append(current_file)
        imports = get_imported_modules(current_file)
        for node in imports:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    files = resolve_module_to_file(alias.name, 0, root_dir)
                    for file in files:
                        if file not in imported_files and file not in files_to_check:
                            files_to_check.append(file)
            elif isinstance(node, ast.ImportFrom):
                files = resolve_module_to_file(node.module, node.level, root_dir)
                for file in files:
                    if file not in imported_files and file not in files_to_check:
                        files_to_check.append(file)
    return imported_files


def detect_encoding(filename):
    """
    Detect the encoding of a file
    """
    with open(filename, "rb") as file:
        rawdata = file.read()
    return chardet.detect(rawdata)["encoding"]



def ingest_directory_contents(root_dir, include_tests=False):
    files_content = {}
    for relative_path in list_files(root_dir, include_tests=include_tests):
        filename = os.path.join(root_dir, relative_path)
        encoding = detect_encoding(filename)
        if encoding is None:
            content = "[BINARY DATA FILE]"
        else:
            try:
                with open(filename, encoding=encoding) as file:
                    content = file.read()
            except (UnicodeDecodeError, LookupError):
                content = "[BINARY DATA FILE]"
        files_content[relative_path] = content
    return files_content


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def convert_files_map_into_list(file_map):
    files_list = []
    for filename in file_map:
        files_list.append(
            {
                "file": filename,
                "content": file_map[filename],
            }
        )
    return files_list


def convert_files_list_into_map(file_list):
    files_map = {}
    for item in file_list:
        files_map[item["file"]] = item["content"]
    return files_map


def convert_components_map_into_list(file_components_map):
    new_components = []

    for file in file_components_map:
        new_components.append(
            {
                "file": file,
                "components": file_components_map[file],
            }
        )
    return new_components


def convert_components_list_to_map(file_components_list):
    new_components = {}
    for item in file_components_list:
        new_components[item["file"]] = item["components"]
    return new_components


