"""Generate settings .md files."""

# Any changes to the overall structure need to be reflected in mkdocs.yml nav section.

import re
from pathlib import Path

from tqdm import tqdm

import mne_bids_pipeline._config

root_file = Path(mne_bids_pipeline._config.__file__)
settings_dir = Path(__file__).parent

# Mapping between first two lower-case words in the section name and the desired
# file or folder name
section_to_file = {  # .md will be added to the files
    # root file
    "general settings": "general",
    # folder
    "preprocessing": "preprocessing",
    "break detection": "breaks",
    "bad channel": "autobads",
    "maxwell filter": "maxfilter",
    "filtering": "filter",
    "resampling": "resample",
    "epoching": "epochs",
    "filtering &": None,  # just a header
    "artifact removal": None,
    "stimulation artifact": "stim_artifact",
    "ssp, ica,": "ssp_ica",
    "amplitude-based artifact": "artifacts",
    # folder
    "sensor-level analysis": "sensor",
    "condition contrasts": "contrasts",
    "decoding /": "mvpa",
    "time-frequency analysis": "time_freq",
    "group-level analysis": "group_level",
    # folder
    "source-level analysis": "source",
    "bem surface": "bem",
    "source space": "forward",
    "inverse solution": "inverse",
    # folder
    "reports": "reports",
    "report generation": "report_generation",
    # root file
    "execution": "execution",
}

option_header = """\
::: mne_bids_pipeline._config
    options:
      members:"""
prefix = """\
        - """

# We cannot use ast for this because it doesn't preserve comments. We could use
# something like redbaron, but our code is hopefully simple enough!
assign_re = re.compile(
    # Line starts with annotation syntax (name captured by the first group).
    r"^(\w+): "
    # Then the annotation can be ...
    "("
    # ... a standard assignment ...
    ".+ = .+"
    # ... or ...
    "|"
    # ... the start of a multiline type annotation like "a: Union["
    r"(Union|Optional|Literal)\["
    # To the end of the line.
    ")$",
    re.MULTILINE,
)


def main():
    print(f"Parsing {root_file} to generate settings .md files.")
    # max file-level depth is 2 even though we have 3 subsection levels
    levels = [None, None]
    current_path, current_lines = None, list()
    text = root_file.read_text("utf-8")
    lines = text.splitlines()
    lines += ["# #"]  # add a dummy line to trigger the last write
    in_header = False
    for li, line in enumerate(tqdm(lines)):
        line = line.rstrip()
        if line.startswith("# #"):  # a new (sub)section / file
            this_def = line[2:]
            this_level = this_def.split()[0]
            assert this_level.count("#") == len(this_level), this_level
            this_level = this_level.count("#") - 1
            if this_level == 2:
                # flatten preprocessing/filtering/filter to preprocessing/filter
                # for example
                this_level = 1
            assert this_level in (0, 1), (this_level, this_def)
            this_def = this_def[this_level + 2 :]
            levels[this_level] = this_def
            # Write current lines and reset
            if len(current_lines) > 1:  # more than just the header
                assert current_path is not None, levels
                if current_lines[0] == "":  # this happens with tags
                    current_lines = current_lines[1:]
                current_path.write_text("\n".join(current_lines + [""]), "utf-8")
            if this_level == 0:
                this_root = settings_dir
            else:
                this_root = settings_dir / f"{section_to_file[levels[0].lower()]}"
            this_root.mkdir(exist_ok=True)
            key = " ".join(this_def.split()[:2]).lower()
            if key == "":
                assert li == len(lines) - 1, (li, line)
                continue  # our dummy line
            fname = section_to_file[key]
            if fname is None:
                current_path = None
            else:
                current_path = this_root / f"{fname}.md"
            current_lines = []
            in_header = True
            continue

        if in_header:
            if line == "":
                in_header = False
                if current_lines:
                    current_lines.append("")
                current_lines.append(option_header)
            else:
                assert line == "#" or line.startswith("# "), (li, line)  # a comment
                current_lines.append(line[2:])
            continue

        # Could be an option
        match = assign_re.match(line)
        if match is not None:
            name, typ, desc = match.groups()
            current_lines.append(f"{prefix}{name}")
            continue


if __name__ == "__main__":
    main()
