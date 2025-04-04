from setuptools import setup, find_packages

setup(
    name="qd_rdfl",
    version="0.1.0",
    description="Data Assetization via Resources-decoupled Federated Learning",
    author="",
    author_email="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "matplotlib",
        "scipy",
    ],
) 