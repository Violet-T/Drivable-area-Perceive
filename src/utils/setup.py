from setuptools import setup

package_name = "freespace_utils"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="perceive",
    maintainer_email="dev@example.com",
    description="Shared image, mask, and geometry utilities for free-space perception.",
    license="MIT",
)
