import tensorflow as tf
import numpy as np
import re

assert tf.__version__ == "1.8.0"

def parse_TF(var_name, tf_str = "", pattern='tf.tensor', shape = None, dtype = None):
    if pattern == 'constant':
        if isinstance(tf_str, str):
            return f"{var_name} = \"{tf_str}\""
        return f"{var_name} = {tf_str}"
    
    shape_str=dtype_str=""
    if shape:
        shape_str = str(shape)
        dtype_str = "float64" # not consider so far
    else:
        shape_match = re.search(r'shape=\((.*?)\)', tf_str)
        dtype_match = re.search(r'dtype=(\w+)', tf_str)
        if shape_match and dtype_match:
            shape_str = shape_match.group(1)
            dtype_str = dtype_match.group(1)

            # Convert shape and dtype to valid Python code
            shape_str = shape_str.replace("?", "None")
        #if shape_match is None:
        #    shape_str = "None"
        #    dtype_str = dtype_match.group(1)
    if len(shape_str)>0 and len(dtype_str)>0:
        if pattern == 'tf.tensor':
            assignment_code = f"{var_name} = tf.placeholder(tf.{dtype_str}, [{shape_str}])"
        elif pattern == 'tf.Variable':
            dtype_str = dtype_str.replace("_ref", "")
            assignment_code = f"{var_name} = tf.Variable(tf.zeros([{shape_str}]))"
        elif pattern == 'np.ndarray': # to initialize as np.random.normal since Pythia doesn't model np.ndarray
            assignment_code = f"{var_name} = np.random.normal(size={shape_str})"
        return assignment_code
    return None

def runtime_info_to_code(who_df, who_ls):
    runtime_code = []
    for item in who_ls:
        if isinstance(who_df[item],tf.Variable):
            assignment_code = parse_TF(item, tf_str=str(who_df[item]), pattern='tf.Variable')
        elif isinstance(who_df[item],tf.Tensor):
            assignment_code = parse_TF(item, tf_str=str(who_df[item]), pattern='tf.tensor')
        elif isinstance(who_df[item],np.ndarray):
            assignment_code = parse_TF(item, shape=str(who_df[item].shape), dtype=str(who_df[item].dtype), pattern='np.ndarray')
        elif isinstance(who_df[item],int) or isinstance(who_df[item],float): # or isinstance(who_df[item],str):
            assignment_code = parse_TF(item, tf_str=who_df[item], pattern='constant')
        else:
            assignment_code = None
        if assignment_code:
            runtime_code.append(assignment_code)
    return '\n'.join(runtime_code)


# alternative to transfer %whos to python code through outputs, men stuck on parsing lines (split separaters)
# run in jupyter environment

# from IPython.utils import capture
# 
# # Capture the output of %whos
# with capture.capture_output() as captured:
#     %whos
#     
# # Function to convert %whos output into assignment Python code
# def whos_to_assignment(whos_output):
#     assignments = []
#     lines = whos_output.strip().split()[2:]  # Skip the header line
# 
#     for line in lines:
#         parts = line.split() # use what separater???
#         if len(parts) == 3:
#             var_name, var_type, var_value = parts
#             if var_type in ['int', 'float32', 'str', 'tuple', 'Variable', 'Tensor']:
#                 assignments.append(f"{var_name} = {var_value}")
# 
#     return '\n'.join(assignments)
# 
# # Convert %whos output to assignment code
# assignment_code = whos_to_assignment(captured.stdout)