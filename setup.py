from setuptools import setup

setup(
    name="constant-acceleration-flow",
    py_modules=["flow", "evaluations"],
    install_requires=[
        "tqdm",
        "numpy<2",
        "scipy",
        "pandas",
        "Cython",
        "piq==0.7.0",
        "joblib==0.14.0",
        "albumentations==0.4.3",
        "pillow",
        "accelerate==0.24.1",
        "einops",
        "wandb",
        "ema-pytorch",
        "xformers==0.0.24",
        "timm==0.5.4",
        "ftfy==6.1.1",
        "regex",
        "tensorflow",
    ],
)