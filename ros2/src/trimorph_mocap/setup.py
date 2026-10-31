from setuptools import setup

setup(
    name="trimorph_mocap",
    version="0.0.1",
    packages=["trimorph_mocap"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/trimorph_mocap"]),
        ("share/trimorph_mocap", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "scipy", "matplotlib>=3.3"],
    zip_safe=False,
    maintainer="TRIMORPH maintainers",
    maintainer_email="local@example.com",
    description="TRIMORPH mocap subscription, recording and visualization",
    license="UNLICENSED",
    entry_points={
        "console_scripts": [
            "mocap_detection = trimorph_mocap.mocap_detection:cli",
        ],
    },
)
