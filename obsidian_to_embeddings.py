import os
import sys
from markdown import Markdown
from io import StringIO
from pathlib import Path

OBSIDIAN_VAULT_PATH = '"<<<path_to_a_obsidian_vault_folder_or_subfolder (e.g. /Users/unixuser/obsidian_vaults)>>>"'
CONTENT_LEN_PER_ITEM_THRESHOLD = 500

def unmark_element(element, stream=None):
    if stream is None:
        stream = StringIO()
    if element.text:
        stream.write(element.text)
    for sub in element:
        unmark_element(sub, stream)
    if element.tail:
        stream.write(element.tail)
    return stream.getvalue()

def unmark(text):
    return __md.convert(text)

# patching Markdown
Markdown.output_formats["plain"] = unmark_element
__md = Markdown(output_format="plain")
__md.stripTopLevelTags = False

def parse(should_print):
    output_content_list = []
    for subdir, dirs, filenames in os.walk(OBSIDIAN_VAULT_PATH):
        for filename in filenames:
            if not filename.endswith('.md'): 
                continue
            f = open(os.path.join(subdir, filename), "r")
            current_line = f.readline()
            accumulator = []
            while current_line:
                current_line = f.readline()
                if ("![[./_resources/" in current_line):
                    # Skip resource references
                    continue
                filename_without_suffix = Path(filename).with_suffix('')
                formatted_line = f'{unmark(current_line)}' 
                if (formatted_line.strip() == ""):
                    continue
                if (should_print):
                    print(formatted_line)
                accumulator.append(formatted_line)
                accumulated_lines = "\n".join(accumulator)
                if (len(accumulated_lines) > CONTENT_LEN_PER_ITEM_THRESHOLD):
                    output_content_list.append((accumulated_lines, f'<obsidian>:{filename_without_suffix}'))
                    accumulator = []
    return output_content_list

def main():
    parse(False)
    return 0

if __name__ == '__main__':
    sys.exit(main())
