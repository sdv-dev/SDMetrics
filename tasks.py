import glob
import os
import shutil
import stat
from pathlib import Path

from invoke import task


@task
def pytest(c):
    c.run('python -m pytest --reruns 5 --cov=sdmetrics')


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


@task
def tutorials(c):
    for ipynb_file in glob.glob('tutorials/*.ipynb') + glob.glob('tutorials/**/*.ipynb'):
        if '.ipynb_checkpoints' not in ipynb_file:
            c.run((
                'jupyter nbconvert --execute --ExecutePreprocessor.timeout=3600 '
                f'--to=html --stdout {ipynb_file}'
            ), hide='out')


@task
def lint(c):
    c.run('flake8 sdmetrics tests')
    c.run('isort -c --recursive sdmetrics tests')


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass
