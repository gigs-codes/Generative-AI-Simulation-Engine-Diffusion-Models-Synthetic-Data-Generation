from setuptools import setup, find_packages

setup(
    name="genai-simulation-engine",
    version="1.0.0",
    description="Generative AI Simulation Engine with Diffusion Models for Synthetic Data Generation",
    author="GenAI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "accelerate>=0.21.0",
        "diffusers>=0.21.0",
        "transformers>=4.33.0",
        "einops>=0.6.1",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "omegaconf>=2.3.0",
        "wandb>=0.15.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "isort"],
        "dashboard": ["flask", "flask-socketio", "plotly"],
        "export": ["onnx", "onnxruntime"],
    },
    entry_points={
        "console_scripts": [
            "genai-train=scripts.train:main",
            "genai-generate=scripts.generate:main",
            "genai-evaluate=scripts.evaluate:main",
        ]
    },
)
