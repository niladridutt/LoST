from setuptools import setup, find_packages


setup(
    name="direct3d",
    version="1.0.0",
    description="Direct3D: Scalable Image-to-3D Generation via 3D Latent Diffusion Transformer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "numpy",
        "cython",
        "trimesh",
        "diffusers",
    ],
)