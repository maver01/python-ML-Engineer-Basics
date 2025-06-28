# python-ML-Engineer-Basics

Python tutorials from Data Structures to ML Engineering. Each topic is in a standalone folder. It use poetry for library management. All commands assume Ubuntu 22.4 OS.

Topics:

1. Python Core Skills
2. Python Data Structures and Algorithms
3. Test Driven Development (TDD)
4. Containers and Linux
5. Databases
6. Backend
7. Cloud concepts
8. MLOps and Model Lifecycle
9. Machine Learning Libraries
10. Distributed Systems

## Local setup

Install pyenv using this guide: https://realpython.com/intro-to-pyenv/

Install dependencies:

```
$ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl
```

Install pyenv:

```
$ curl https://pyenv.run | bash
```

Install python:

```
pyenv install -v 3.11.2
```

See available installed python version:

```
pyenv versions
```

Activate a python version:

```
pyenv local 3.11.2
```

Now python is installed, and ready to be used to run code.

Install pipx from this guide: https://pipx.pypa.io/stable/installation/.

```
sudo apt update
sudo apt install pipx
pipx ensurepath
```

Install poetry from this guide: https://python-poetry.org/docs/

```
pipx install poetry
```

Now poetry is installed and ready to be used to install venv and libraries.
