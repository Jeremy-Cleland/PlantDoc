from setuptools import find_namespace_packages, setup

setup(
    name="core",
    version="0.1.0",
    packages=find_namespace_packages(include=["*"]),
    entry_points={
        "console_scripts": [
            "core=core.cli:main",
        ],
    },
)
