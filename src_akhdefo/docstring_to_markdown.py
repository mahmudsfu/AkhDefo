import ast
import os
import argparse

def docstring_to_markdown(py_file_path):
    """
    Extract docstrings from a Python file and save them in a Markdown file.

    :param py_file_path: Path to the Python file.
    """
    with open(py_file_path, "r") as file:
        tree = ast.parse(file.read())

    markdown_lines = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Extract the name and the docstring
            name = node.name
            docstring = ast.get_docstring(node)
            if docstring:
                # Format as Markdown
                markdown_lines.append(f"## {name}\n")
                markdown_lines.append(f"{docstring}\n")

    # Define the output Markdown file path
    md_file_path = py_file_path.replace(".py", ".md")
    
    # Save the extracted docstrings to a Markdown file
    with open(md_file_path, "w") as md_file:
        md_file.write("\n".join(markdown_lines))

    print(f"Markdown file saved to {md_file_path}")

def process_directory(directory):
    """
    Process each Python file in the given directory to generate Markdown documentation.

    :param directory: Directory containing Python files.
    """
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            py_file_path = os.path.join(directory, filename)
            docstring_to_markdown(py_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Markdown documentation from Python file docstrings.")
    parser.add_argument("dir", type=str, help="Directory containing Python files.")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.dir):
        process_directory(args.dir)
    else:
        print(f"The specified directory {args.dir} does not exist or is not a directory.")
