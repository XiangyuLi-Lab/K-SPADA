from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as readme_file:
    long_description = readme_file.read()


with open("requirements.txt", "r", encoding="utf-8") as requirements_file:
    install_requires = [
        "torch==2.1.1" if line.strip().startswith("torch==") else line.strip()
        for line in requirements_file
        if line.strip() and not line.startswith("#") and not line.startswith("--")
    ]


setup(
    name="kspada",
    version="0.2.0",
    description="Organized K-SPADA model, training, prediction, data processing, and tutorials.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["KSPADA", "KSPADA.*"]),
    include_package_data=True,
    package_data={"KSPADA": ["resources/*.gmt", "resources/*.txt"]},
    python_requires=">=3.8",
    install_requires=install_requires,
)
