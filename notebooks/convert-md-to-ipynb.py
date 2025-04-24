import os
import re
import nbformat as nbf
import argparse

def remove_challenge_solutions(md_content):
    """
    Removes blocks starting with '> ## Solution' until the end of the block or another challenge block.
    """
    lines = md_content.splitlines()
    result = []
    skip = False
    for line in lines:
        if re.match(r'>\s*##+\s*Solution', line):
            skip = True
            continue
        if skip and line.strip().startswith('{:.challenge}') or line.strip() == '':
            skip = False
            continue
        if not skip:
            result.append(line)
    return '\n'.join(result)

def md_to_notebook(md_file, notebook_file, base_image_url):
    with open(md_file, 'r') as file:
        md_lines = file.readlines()

    print("Preview of the first 100 lines of the Markdown file:")
    for i, line in enumerate(md_lines[:100]):
        print(f"{i+1}: {line}", end='')
    print("\n")

    md_content = ''.join(md_lines)

    nb = nbf.v4.new_notebook()
    cells = []

    # Title from YAML
    title_match = re.search(r'^---\s*title:\s*"(.*?)".*?---', md_content, flags=re.DOTALL | re.MULTILINE)
    if title_match:
        title = title_match.group(1)
        cells.append(nbf.v4.new_markdown_cell(f"# {title}"))

    # Remove YAML front matter
    md_content = re.sub(r'^---.*?---', '', md_content, flags=re.DOTALL | re.MULTILINE)

    # Remove > ## Solution blocks
    md_content = remove_challenge_solutions(md_content)

    # Replace image paths
    def replace_image_path(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        web_image_url = f"{base_image_url}/{os.path.basename(image_path)}"
        return f"![{alt_text}]({web_image_url})"

    md_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image_path, md_content)

    # Split into lines
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

    process_buffer(text_buffer, "markdown")
    cells = [cell for cell in cells if cell['source'].strip()]
    nb['cells'] = cells

    with open(notebook_file, 'w') as file:
        nbf.write(nb, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown file to Jupyter Notebook")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    parser.add_argument("base_image_url")
    parser.add_argument("filename")

    args = parser.parse_args()

    input_path = os.path.join(args.input_dir, args.filename)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, os.path.splitext(args.filename)[0] + ".ipynb")

    print(f"Converting {input_path} to {output_path}...")
    md_to_notebook(input_path, output_path, args.base_image_url)
