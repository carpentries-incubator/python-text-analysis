import os
import re
import nbformat as nbf
import argparse


def md_to_notebook(md_file, notebook_file, base_image_url):
    # Read the Markdown file
    with open(md_file, 'r') as file:
        md_lines = file.readlines()

    # Preview the first 100 lines of the Markdown file
    print("Preview of the first 100 lines of the Markdown file:")
    for i, line in enumerate(md_lines[:100]):
        print(f"{i+1}: {line}", end='')
    print("\n")

    md_content = ''.join(md_lines)

    # Initialize a new notebook
    nb = nbf.v4.new_notebook()
    cells = []

    # Extract the title from the YAML front matter
    title_match = re.search(r'^---\s*title:\s*"(.*?)".*?---', md_content, flags=re.DOTALL | re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        cells.append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Remove the YAML front matter
    md_content = re.sub(r'^---.*?---', '', md_content, flags=re.DOTALL | re.MULTILINE)

    # Replace local image paths with URLs
    def replace_image_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        web_image_url = f"{base_image_url}/{os.path.basename(image_path)}"
        return f"![{alt_text}]({web_image_url})"

    md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, md_content)

    # Split the content into lines
    lines = md_content.split('\n')

    code_buffer = []
    is_in_code_block = False
    text_buffer = []

    def process_buffer(buffer, cell_type="markdown"):
        if buffer:
            content = '\n'.join(buffer).strip()
            if cell_type == "code":
                cells.append(nbf.v4.new_code_cell(content))
            else:
                cells.append(nbf.v4.new_markdown_cell(content))
            buffer.clear()

    for line in lines:
        if line.startswith('```'):
            is_in_code_block = not is_in_code_block
            if is_in_code_block:
                process_buffer(text_buffer, "markdown")
            else:
                process_buffer(code_buffer, "code")
        elif is_in_code_block:
            code_buffer.append(line)
        else:
            text_buffer.append(line)

    # Process any remaining buffers
    process_buffer(text_buffer, "markdown")

    # Remove empty cells
    cells = [cell for cell in cells if cell['source'].strip()]

    # Add cells to the notebook
    nb['cells'] = cells

    # Write the notebook to a file
    with open(notebook_file, 'w') as file:
        nbf.write(nb, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown file to Jupyter Notebook")
    parser.add_argument("input_dir", help="Directory containing the input Markdown file")
    parser.add_argument("output_dir", help="Directory to save the output Jupyter Notebook")
    parser.add_argument("base_image_url", help="Base URL for images")
    parser.add_argument("filename", help="Filename of the input Markdown file (with extension)")

    args = parser.parse_args()

    # Ensure directories have trailing slashes and exist
    input_path = os.path.join(args.input_dir, args.filename)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.splitext(args.filename)[0] + ".ipynb")

    print(f"Converting {input_path} to {output_path}...")

    md_to_notebook(input_path, output_path, args.base_image_url)

#e.g., ...
# python convert_md_to_notebook.py ./episodes ./notebooks https://github.com/user/repo/raw/main/images example.md

