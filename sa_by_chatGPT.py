import tokenize
import io
import os
import openai
from openai import OpenAI
import time
import re
client = OpenAI()
 
# define a retry decorator
def retry_with_suggested_backoff(
    func,
    max_retries: int = 10,
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except openai.RateLimitError as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                
                pattern = r"Please try again in (\d+(\.\d+)?(?:s|ms|m))"
                match = re.search(pattern, str(e))
                if match:
                    time_str = match.group(1)
                    # Convert the time value to seconds
                    if time_str.endswith('ms'):
                        seconds = float(time_str[:-2]) / 1000
                    elif time_str.endswith('s'):
                        seconds = float(time_str[:-1])
                    elif time_str.endswith('m'):
                        seconds = float(time_str[:-1]) * 60
                    else:
                        print(f"Rate limit reached, failed to extract time: {time_str}.")
                        seconds = 1
                    delay = seconds+1
                    print(f"Rate limit reached, retry in {delay}s.")
                else:
                    delay = 10
                    print(f"Rate limit reached, but no retry time suggested: {e}. Retry in {delay}s.")
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper
    
@retry_with_suggested_backoff
def request_chatGPT_with_backoff(tar_code, max_tokens = 4096):
    completion = client.chat.completions.create(
        model = "gpt-4", #"gpt-3.5-turbo", #
        messages = [
            {
                "role": "system",
                "content": "You will be given a piece of python code, many of which use tensorflow library. Your job is to find shape related errors if there are any. Only focus on errors that occur when the used shape does not match the expected shape in an operator or function. Ignore the errors related to redefine or fixed shape definitions. Generate outputs according to the templates: \"No shape mismatch found.\" if no shape errors found, or \"[error]: [LINENUMBER: LINEOFCODE] REASON\" if found any shape errors and make sure to provide reasons."
            },
            {
                "role": "user",
                "content": tar_code
            }
        ],
        temperature = 0,
        max_tokens = max_tokens,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )
    res = completion.choices[0].message.content
    return res

def remove_comments_and_docstrings_python(file_path, target_folder):
    with open(file_path, 'r') as file:
        source = file.read()
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        ltext = tok[4]
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string
        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line
    out = '\n'.join(l for l in out.splitlines() if l.strip())
    file_new = open(os.path.join(target_folder, os.path.basename(file_path)), "w")
    file_new.write(out)
    file_new.close()

def get_py_paths(folder):
    res_file_paths = {
        "buggy": [],
        "buggy_runinfo":[],
        "fix": [],
        "fix_runinfo": []
    }

    folder_buggy = os.path.join(folder, 'tf_bugs_py')
    for file in os.listdir(folder_buggy):
        if file.startswith('ut') and file.endswith('.py'):
            if file.endswith('_runinfo.py'):
                res_file_paths["buggy_runinfo"].append(os.path.join(folder_buggy, file))
            else:
                res_file_paths["buggy"].append(os.path.join(folder_buggy, file))
    folder_fix = os.path.join(folder, 'tf_fix_py')
    for file in os.listdir(folder_fix):
        if file.startswith('ut') and file.endswith('.py'):
            if file.endswith('_runinfo.py'):
                res_file_paths["fix_runinfo"].append(os.path.join(folder_fix, file))
            else:
                res_file_paths["fix"].append(os.path.join(folder_fix, file))

    return res_file_paths

def chatGPT_py_file(file_path):
    tar_code = ''
    with open(file_path, 'r') as file:
        tar_code = file.read()
    #print(tar_code)
    if len(tar_code)>0:
        res = request_chatGPT_with_backoff(tar_code)
    else:
        res='The target source code file is empty.'
    return res