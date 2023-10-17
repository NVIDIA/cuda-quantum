import sys
import re
import glob
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def available():
    available_backends = sys.stdin.readlines()

    if available_backends:
        available_backends = [available_backend.strip()
                              for available_backend in available_backends]
        return available_backends


def validate(notebook_filename):

    with open(notebook_filename) as f:
        lines = f.readlines()
    for notebook_content in lines:
        status = True
        match = re.search('set_target[\\\s\(]+"(.+)\\\\"[)]', notebook_content)
        if match:
            if (match.group(1) not in available_backends):
                status = False
                break
    return status


def execute(notebook_filename, run_path='.'):
    notebook_filename_out = notebook_filename.replace(
        '.ipynb', '.nbconvert.ipynb')

    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    success = True
    try:
        out = ep.preprocess(nb, {'metadata': {'path': run_path}})
    except CellExecutionError:
        out = None
        msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
        msg += 'See notebook "%s" for the traceback.' % notebook_filename_out
        print(msg)
        success = False
        raise
    finally:
        with open(notebook_filename_out, mode='w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        return success


if __name__ == "__main__":

    available_backends = available()
    notebook_filenames = glob.glob(
        f"docs/sphinx/examples/python/tutorials/*.ipynb")
    notebook_filenames = [
        fn for fn in notebook_filenames if not fn.endswith('.nbconvert.ipynb')]

    if notebook_filenames:
        notebooks_failed = []
        for notebook_filename in notebook_filenames:
            if (validate(notebook_filename)):
                notebooks_failed.append(notebook_filename)
            else:
                print(f"Skipped! Missing backends for {notebook_filename}")

        if len(notebooks_failed) > 0:
            print("Failed! The following notebook(s) raised errors:\n" +
                  " ".join(notebooks_failed))
        else:
            print("Success! All the notebook(s) executed successfully.")
    else:
        print('Failed! No notebook found in the current directory.')
