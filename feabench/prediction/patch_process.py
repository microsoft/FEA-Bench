import re
from difflib import SequenceMatcher, unified_diff
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Constants - Miscellaneous
NON_TEST_EXTS = [".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml"]

def is_test_file(path):
    if any(
        test_word in path for test_word in
        ['test', 'tests', 'e2e', 'testing', 'check']
    ):
        if not any(path.endswith(ext) for ext in NON_TEST_EXTS):
            return True
    
    return False


def find_matches(file_lines, lines):
    aligned_idx = []
    for i in range(len(file_lines) - len(lines) + 1):
        slice = file_lines[i:i+len(lines)]
        if slice == lines:
            aligned_idx.append(i)
    return aligned_idx


def parse_hunk_header(hunk_header):
    """
    Parses a unified diff hunk header and extracts the start line numbers and lengths for both old and new files.
    
    Args:
        hunk_header (str): The hunk header line in the format '@@ -X,Y +A,B @@'
        
    Returns:
        tuple: A tuple containing four integers (old_start, old_length, new_start, new_length)
    """
    extract_by_split = hunk_header.strip().split("@@")
    
    pattern = r'@@ -(?P<old_start>\d+)(?:,(?P<old_length>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_length>\d+))? @@'

    pure_header = "@@" + extract_by_split[1] + "@@"
    match = re.match(pattern, pure_header.strip())
    if not match:
        raise ValueError("Invalid hunk header format")

    old_start = int(match.group('old_start'))
    old_length = int(match.group('old_length')) if match.group('old_length') else 1
    new_start = int(match.group('new_start'))
    new_length = int(match.group('new_length')) if match.group('new_length') else 1

    return (old_start, old_length, new_start, new_length)


def parse_segments(text):
    """Parse the segments into a list of tuples containing file paths and code changes."""
    # pattern = r'\[start of segment in (\S+)\]([\s\S]*?)\[end of segment in \1\]'
    pattern = re.compile(
        r'\[start of the snippet (before|after) editing in (?P<file>[\s\S]*?)\](?P<code>.*?)\[end of the snippet \1 editing in (?P=file)\]',
        re.DOTALL
    )
    matches = re.finditer(pattern, text)
    segments = []
    for match in matches:
        file_path = match.group('file').strip()
        content = match.group('code').split('\n')
        if match.group('code').strip():
            # remove blank line at front or back
            start_index, end_index = 0, len(content) - 1
            while content[start_index].strip() == "":
                start_index += 1
            while content[end_index].strip() == "":
                end_index -= 1
            content = content[start_index: end_index+1]
        segments.append((file_path, content))
    
    if len(segments) % 2 != 0:
        logger.info("[SKIP patch last][Error] There is unmatched edit file. This may be caused by truncation in the output.")
        segments = segments[:-1]

    if len(segments) == 0:
        logger.info("[SKIP instance][Error] There is no complete edit. This may be caused by truncation in the output.")

    return segments



def create_patch_by_compare_file(segments, files):
    """Create a unified diff patch from parsed segments."""
    patches_info = {}
    
    # Group segments by file path
    for i in range(0, len(segments), 2):
        old_file, old_content = segments[i]
        new_file, new_content = segments[i + 1]

        # file name
        file = new_file
        if file.startswith(("a/", "b/")):
            file = file[2:]
        assert old_file.endswith(file), f"[SKIP instance][Error] The file name is not matched in {old_file} and {new_file}"
        
        if file not in files:
            # discard if there is a code snippet which does not come from context files
            logger.info(f"[SKIP patch][Warning] There is no corresponding file in the context for {file}")
            continue

        # extract patch info
        if files[file]:
            # the file exists
            file_content = files[file].splitlines()

            # remove the space at right
            file_content_for_match = [l.rstrip() for l in file_content]
            old_content_for_match = [l.rstrip() for l in old_content]
            new_content_for_match = [l.rstrip() for l in new_content]

            aligned_indices = find_matches(file_content_for_match, old_content_for_match)

            # calc differ start offset
            differ_start_offset = 0
            max_len = min(len(old_content_for_match), len(new_content_for_match))
            while differ_start_offset < max_len and new_content_for_match[differ_start_offset] == old_content_for_match[differ_start_offset]:
                differ_start_offset += 1

            # if there is no difference continue
            if differ_start_offset == max_len:
                logger.info(f"[SKIP patch][Warning] There is no difference between the code snippet {i} and {i+1}")
                continue

            # calc differ end offset
            differ_end_offset = -1
            while differ_end_offset > - max_len and new_content_for_match[differ_end_offset] == old_content_for_match[differ_end_offset]:
                differ_end_offset -= 1
            
            if len(aligned_indices) != 1:
                matched_lines = [no + 1 for no in aligned_indices]
                logger.info(f"[SKIP patch][Error]The {i} th content of generated segments cannot be aligned with original file bacause the aligned line includes: {matched_lines}")
                continue

            patch_item = {
                "file_name": file,
                "start_no": aligned_indices[0] + 1,
                "first_different_line": aligned_indices[0] + 1 + differ_start_offset,
                "last_different_line": aligned_indices[0] + 1 + len(old_content_for_match) + differ_end_offset,
                "differ_start_offset": differ_start_offset,
                "differ_end_offset": differ_end_offset,
                "old": old_content,
                "new": new_content,
            }

            if file not in patches_info:
                patches_info[file] = [patch_item]
            else:
                patches_info[file].append(patch_item)
        else:
            # new file
            assert "".join(old_content).strip() == "", f"[SKIP patch][Error] The new created file {file} should not contain any content before editing."
            patch_item = {
                "file_name": file,
                "start_no": 0,
                "first_different_line": 0,
                "last_different_line": 0,
                "differ_start_offset": 0,
                "differ_end_offset": 0,
                "old": None,
                "new": new_content,
            }
            if file not in patches_info:
                patches_info[file] = [patch_item]
            else:
                patches_info[file].append(patch_item)

    # get string patches
    patch_all = []

    for file_path, data in patches_info.items():
        # ignore the test file edit
        if is_test_file(file_path):
            logger.info(f"[SKIP file][Warning] The diff for test file will not be included in the patch.")
            continue

        data = sorted(data, key=lambda x:x["first_different_line"], reverse=True)
        diff_in_file = []
        file_header = None

        # the original file
        file_content = files[file_path].splitlines() if files[file_path] else []

        # net added line num due to the edit
        last_applyed_start_index = len(file_content)

        for patch_item in data:
            ## process with new file
            if patch_item["old"] == None:
                # new file
                patch_all += [
                    f"diff --git a/{file_path} b/{file_path}",
                    "new file mode 100644",
                    "--- /dev/null",
                    f"+++ b/{file_path}",
                    f"@@ -0,0 +1,{len(patch_item['new'])} @@",
                ]
                patch_all += ["+" + l for l in patch_item["new"]]
                break

            ## for existing file, apply changes
            if last_applyed_start_index <= patch_item["last_different_line"] - 1:
                logger.info(f"[SKIP block][Error] There is overlap changed span.")
                continue
            pure_new_content = patch_item["new"][patch_item["differ_start_offset"] : patch_item["differ_end_offset"] + 1 + len(patch_item["new"])]
            file_content = file_content[ :patch_item["first_different_line"] - 1] + pure_new_content + file_content[patch_item["last_different_line"]: ]
            last_applyed_start_index = patch_item["first_different_line"] - 1

        # compute diff if changes exist
        if file_content:
            diff = list(unified_diff(
                files[file_path].splitlines(), file_content,
                fromfile=file_path, tofile=file_path,
                lineterm=''
            ))
            # Remove unnecessary headers added by unified_diff
            file_header = [
                f"diff --git a/{file_path} b/{file_path}",
                f"--- a/{file_path}",
                f"+++ b/{file_path}"
            ]
            patch_all = patch_all + file_header + diff[2:]

    # fix patch unexpectedly ends in middle of line
    if patch_all and patch_all[-1].startswith("+"):
        patch_all.append("")

    # Construct final patch string
    patch_str = "\n".join(patch_all)

    return patch_str


# For generate patch directly        
def extract_diff(response):
    """
    Extracts the diff from a response formatted in different ways
    """
    if response is None:
        return ""
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



example_output = """
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
logger.info(factorial(5))
[end of the snippet after editing in src/demo.py]
</edit>
"""

example_files = {
    "src/code.py": """def factorial(a):
    res = 1
    while a >= 0:
        res *= a
    return res
    #
    # 
    # 
    # 
def exact_dividion(x, y):
    return x % y == 0""",
    "src/application.py": """
# examples of math application
# 
# 
values = [1,2,3,4,5]

num = sum(values)
assert type(num) == int and num>=0
result = factorial(num)

""",
    "src/demo.py": None,
}


# post process
def postprocess(response, data_item, prompt_key):
    if str(prompt_key).startswith("text"):
        patch = extract_diff(response)
        lines = patch.split("\n")
        if lines and lines[-1].startswith("+"):
            patch += "\n"
        return patch
    elif str(prompt_key).startswith("natural"):
        files_map = {
            item["file"]: item["content"]
            for item in data_item["files"]
        }
        segments = parse_segments(response)
        patch = create_patch_by_compare_file(segments, files_map)
        return patch
    else:
        raise NotImplementedError()


def test_natural_example():
    segments = parse_segments(example_output)
    patch = create_patch_by_compare_file(segments, example_files)
    logger.info(patch)


# Example usage
if __name__ == "__main__":
    test_natural_example()

