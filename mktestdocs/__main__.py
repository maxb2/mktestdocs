import inspect
import os
import pathlib
import re
import subprocess
import textwrap
from collections import OrderedDict
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

CLASS_RE = re.compile(
    dedent(
        r"""
        [ \t]*
        \.
        (?P<class>[a-zA-Z][a-zA-Z0-9_\-]*)
        [ \t]*
        """
    ),
    re.DOTALL | re.VERBOSE,
)

ID_RE = re.compile(
    dedent(
        r"""
        [ \t]*
        \#
        (?P<id>[a-zA-Z][a-zA-Z0-9_\-]*)
        [ \t]*
        """
    ),
    re.DOTALL | re.VERBOSE,
)

KEY_VAL_RE = re.compile(
    dedent(
        r"""
        [ \t]*
        (?P<key>\b[a-zA-Z][a-zA-Z0-9_]*)
        (?:
            =
            (?P<quot>"|')
            (?P<value>.*?)
            (?P=quot)
        )?
        [ \t]*
        """
    ),
    re.DOTALL | re.VERBOSE,
)

# NOTE: this is modified from
# `markdown.extensions.fenced_code.FencedBlockPreprocessor.FENCED_BLOCK_RE`
# to include options from `pymdownx/superfences.RE_OPTIONS`
FENCED_BLOCK_RE = re.compile(
    dedent(
        r"""
        (?P<raw>
            (?P<fence>^(?:~{3,}|`{3,}))[ ]*           # opening fence
            (
                (\{(?P<attrs>[^\}\n]*)\})|            # (optional {attrs} or
                (\.?(?P<lang>[\w#.+-]*)[ ]*)?         # optional (.)lang
                (?P<options>                          # optional "options"
                    (?:                               # key-value pairs
                        (?:
                            \b[a-zA-Z][a-zA-Z0-9_]*
                            (?:
                                =
                                (?P<quot>"|')
                                .*?
                                (?P=quot)
                            )?
                            [ \t]*
                        ) |                           
                    )*
                )
            )
            \n                                        # newline (end of opening fence)
            (?P<code>.*?)(?<=\n)                      # the code block
            (?P=fence)[ ]*$                           # closing fence
        )
        """
    ),
    re.MULTILINE | re.DOTALL | re.VERBOSE,
)


class Attr(NamedTuple):
    value: str
    type: str


# @dataclass
class Fence(NamedTuple):
    fence: str = ""
    lang: Optional[str] = None
    attrs: Optional[OrderedDict[str, Attr]] = None
    options: Optional[Dict[str, Any]] = None
    contents: str = ""
    raw: Optional[str] = None

    def options_from_str(raw: str) -> Dict[str, Any]:
        """Markdown fence options dict from string

        Args:
            raw (str): string of options

        Returns:
            Dict[str, Any]: dict of options
        """
        options = {}
        while raw:
            match = KEY_VAL_RE.match(raw)
            if match is None:
                break
            options[match.groupdict()["key"]] = match.groupdict()["value"]
            raw = raw[match.span()[1] :]
        return options

    def attrs_from_str(raw: str) -> OrderedDict[str, Attr]:
        """Markdown fence attributes from string.

        Args:
            raw (str): string of attributes

        Returns:
            Dict[str, Any]: dict of attrs
        """

        attrs = OrderedDict()

        while raw:
            if match := CLASS_RE.match(raw):
                attrs[match.groupdict()["class"]] = Attr(value=None, type="class")
            elif match := ID_RE.match(raw):
                attrs[match.groupdict()["id"]] = Attr(value=None, type="id")
            elif match := KEY_VAL_RE.match(raw):
                attrs[match.groupdict()["key"]] = Attr(
                    value=match.groupdict()["value"], type="keyval"
                )
            else:
                break
            raw = raw[match.span()[1] :]
        return attrs

    def from_re_groups(groups: Tuple[str]) -> "Fence":
        """Make Fence from regex groups

        Notes:
            This is tightly coupled to `FENCED_BLOCK_RE`.

        Args:
            groups (Tuple[str]): regex match groups

        Returns:
            Fence: markdown fence
        """

        attrs = Fence.attrs_from_str(groups[4])

        try:
            lang_attr = list(attrs.items())[0]
            _lang = lang_attr[0] if lang_attr[1].type == "class" else None
        except IndexError:
            _lang = None

        lang = groups[6] or _lang

        return Fence(
            fence=groups[1],
            lang=lang,
            attrs=attrs,
            options=Fence.options_from_str(groups[7]),
            contents=dedent(groups[9]),
            raw=groups[0],
        )

    def from_str(raw: str) -> "Fence":
        """Fence from markdown string

        Args:
            raw (str): markdown string

        Raises:
            Exception: couldn't find a markdown fence

        Returns:
            Fence: markdown fence
        """
        return Fence.from_re_groups(FENCED_BLOCK_RE.match(raw).groups())


_executors = {}


def register_executor(lang, executor):
    """Add a new executor for markdown code blocks

    lang should be the tag used after the opening ```
    executor should be a callable that takes one argument:
        the code block found
    """
    _executors[lang] = executor


def exec_bash(fence: Fence):
    """Exec the bash source given in a new subshell

    Does not return anything, but if any command returns not-0 an error
    will be raised
    """
    command = ["bash", "-e", "-u", "-c", fence.contents]
    try:
        subprocess.run(command, check=True)
    except Exception:
        print(fence.contents)
        raise


register_executor("bash", exec_bash)


def exec_python(fence: Fence, __globals: Optional[Dict] = None):
    """Exec the python source given in a new module namespace

    Does not return anything, but exceptions raised by the source
    will propagate out unmodified
    """
    if __globals is None:
        __globals = {"__MODULE__": "__main__"}

    try:
        exec(fence.contents, __globals)
    except Exception:
        print(fence.contents)
        raise


register_executor("", exec_python)
register_executor("python", exec_python)


def get_codeblock_members(*classes):
    """
    Grabs the docstrings of any methods of any classes that are passed in.
    """
    results = []
    for cl in classes:
        if cl.__doc__:
            results.append(cl)
        for name, member in inspect.getmembers(cl):
            if member.__doc__:
                results.append(member)
    return [m for m in results if len(grab_code_blocks(m.__doc__)) > 0]


def check_codeblock(block, lang="python"):
    """
    Cleans the found codeblock and checks if the proglang is correct.

    Returns an empty string if the codeblock is deemed invalid.

    Arguments:
        block: the code block to analyse
        lang: if not None, the language that is assigned to the codeblock
    """
    first_line = block.split("\n")[0]
    if lang:
        if first_line[3:] != lang:
            return ""
    return "\n".join(block.split("\n")[1:])


def grab_code_blocks(docstring, lang="python"):
    """
    Given a docstring, grab all the markdown codeblocks found in docstring.

    Arguments:
        docstring: the docstring to analyse
        lang: if not None, the language that is assigned to the codeblock
    """
    docstring = textwrap.dedent(docstring)
    in_block = False
    block = ""
    codeblocks = []
    for idx, line in enumerate(docstring.split("\n")):
        if line.startswith("```"):
            if in_block:
                codeblocks.append(check_codeblock(block, lang=lang))
                block = ""
            in_block = not in_block
        if in_block:
            block += line + "\n"
    return [c for c in codeblocks if c != ""]

def grab_fences(source: str) -> List[Fence]:
    """Grab fences in  markdown

    Args:
        source (str): markdown string

    Returns:
        List[Fence]: list of fences in markdown
    """
    return [Fence.from_re_groups(groups) for groups in FENCED_BLOCK_RE.findall(source)]

def grab_code_blocks(docstring, **kwargs):
    return grab_fences(docstring)

def check_docstring(obj, lang=""):
    """
    Given a function, test the contents of the docstring.
    """
    if lang not in _executors:
        raise LookupError(
            f"{lang} is not a supported language to check\n"
            "\tHint: you can add support for any language by using register_executor"
        )
    executor = _executors[lang]
    for b in grab_code_blocks(obj.__doc__, lang=lang):
        executor(b)


def check_raw_string(raw, lang="python"):
    """
    Given a raw string, test the contents.
    """
    if lang not in _executors:
        raise LookupError(
            f"{lang} is not a supported language to check\n"
            "\tHint: you can add support for any language by using register_executor"
        )
    executor = _executors[lang]
    for b in grab_code_blocks(raw, lang=lang):
        executor(b)


def check_raw_file_full(raw, lang="python"):
    if lang not in _executors:
        raise LookupError(
            f"{lang} is not a supported language to check\n"
            "\tHint: you can add support for any language by using register_executor"
        )
    executor = _executors[lang]
    all_code = ""
    for b in grab_code_blocks(raw, lang=lang):
        all_code = f"{all_code}\n{b}"
    executor(all_code)


def check_md_file(fpath, memory=False, lang="python"):
    """
    Given a markdown file, parse the contents for python code blocks
    and check that each independent block does not cause an error.

    Arguments:
        fpath: path to markdown file
        memory: whether or not previous code-blocks should be remembered
    """
    text = pathlib.Path(fpath).read_text()
    if not memory:
        check_raw_string(text, lang=lang)
    else:
        check_raw_file_full(text, lang=lang)


class WorkingDirectory:
    """Sets the cwd within the context"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.origin = Path().absolute()

    def __enter__(self):
        os.chdir(self.path)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        os.chdir(self.origin)


_executors = {}


def register_executor(lang, executor):
    """Add a new executor for markdown code blocks

    lang should be the tag used after the opening ```
    executor should be a callable that takes one argument:
        the code block found
    """
    _executors[lang] = executor


def grab_fences(source: str) -> List[Fence]:
    """Grab fences in  markdown

    Args:
        source (str): markdown string

    Returns:
        List[Fence]: list of fences in markdown
    """
    return [Fence.from_re_groups(groups) for groups in FENCED_BLOCK_RE.findall(source)]


def exec_file_fence(fence: Fence, **kwargs):
    """Executor that writes out file

    Args:
        fence (Fence): markdown fence
    """
    fname = fence.options.get("title", None) or fence.attrs.get("title", [None])[0]
    with open(fname, "w") as f:
        f.write(fence.contents)


register_executor("yaml", exec_file_fence)
register_executor("yml", exec_file_fence)
register_executor("toml", exec_file_fence)


def exec_python_fence(fence: Fence, globals: Dict = {}):
    """Python fence executor

    Args:
        fence (Fence): markdown fence
        globals (Dict, optional): python globals to pass to exec. Defaults to {}.
    """
    if fence.options.get("title", False) or fence.attrs.get("title", False):
        exec_file_fence(fence)
    try:
        exec(fence.contents, globals)
    except Exception:
        print(fence.contents)
        raise


register_executor("python", exec_python_fence)
register_executor("py", exec_python_fence)


def exec_bash_fence(fence: Fence, **kwargs):
    """Bash fence executor

    Args:
        fence (Fence): markdown fence
    """
    _cmds = fence.contents.split("$ ")
    commands: List[Dict] = []
    for _cmd in _cmds:
        if not _cmd:
            continue
        lines = _cmd.splitlines()
        commands.append({"input": lines[0], "output": "\n".join(lines[1:])})

    for command in commands:
        result = run(command["input"], shell=True, check=True, capture_output=True)
        assert result.stdout.decode().strip() == command["output"].strip()


register_executor("bash", exec_bash_fence)



