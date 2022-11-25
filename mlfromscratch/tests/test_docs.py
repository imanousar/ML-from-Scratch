import pathlib
import pytest

from mktestdocs import check_md_file

doc_paths = pathlib.Path("mlfromscratch/docs").glob("**/*.md")

# Note the use of `ids` as it makes for pretty output


@pytest.mark.parametrize('fpath', doc_paths, ids=str)
def test_files_good(fpath):
    check_md_file(fpath=fpath, memory=True)
