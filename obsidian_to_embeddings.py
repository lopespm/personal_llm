import os
import sys
from markdown import Markdown
from io import StringIO
from pathlib import Path
from tqdm import tqdm

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
    all_md_files = [
        (subdir, filename)
        for subdir, _, filenames in os.walk(OBSIDIAN_VAULT_PATH)
        for filename in filenames
        if filename.endswith('.md')
    ]
    output_content_list = []
    for subdir, filename in tqdm(all_md_files, desc='Parsing vault', unit='file'):
        filename_without_suffix = Path(filename).with_suffix('')
        accumulator = []
        accumulated_len = 0
        with open(os.path.join(subdir, filename), "r") as f:
            for current_line in f:
                if "![[./_resources/" in current_line:
                    continue
                formatted_line = unmark(current_line)
                if formatted_line.strip() == "":
                    continue
                if should_print:
                    print(formatted_line)
                accumulator.append(formatted_line)
                accumulated_len += len(formatted_line) + 1  # +1 for the joining \n
                if accumulated_len > CONTENT_LEN_PER_ITEM_THRESHOLD:
                    output_content_list.append(("\n".join(accumulator), f'<obsidian>:{filename_without_suffix}'))
                    accumulator = []
                    accumulated_len = 0
    return output_content_list

def main():
    parse(False)
    return 0

if __name__ == '__main__':
    sys.exit(main())
