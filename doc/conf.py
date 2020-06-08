from pathlib import Path

import sphinx_rtd_theme


def read(filepath):
    import codecs

    with codecs.open(filepath, "r") as fp:
        return fp.read()


def find_version(filepath):
    import re

    version_file = read(filepath)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]

source_suffix = ".rst"

master_doc = "index"

project = "numpy-sugar"
copyright = "2018, Danilo Horta"
author = "Danilo Horta"

version = find_version(Path(__file__).parents[0] / Path("../numpy_sugar/__init__.py"))
release = version

language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "conf.py"]

pygments_style = "default"

todo_include_todos = False

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_sidebars = {"**": ["relations.html", "searchbox.html"]}

htmlhelp_basename = "{}doc".format(project)

man_pages = [(master_doc, project, "{} documentation".format(project), [author], 1)]

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}
