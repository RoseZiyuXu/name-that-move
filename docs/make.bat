@ECHO OFF

set SPHINXBUILD=python -m sphinx
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% -W --keep-going
