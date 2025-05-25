from setuptools import setup, find_packages

setup(
    name="beautyaiinference",
    version="1.0.0",
    description="A modular framework for inference with Arabic AI models",
    author="Lumi AI",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "dev": ["black", "isort", "flake8", "pytest"],
    },
    entry_points={
        "console_scripts": [
            "beautyai-chat=beautyai_inference.cli.chat_cli:main",
            "beautyai-test=beautyai_inference.cli.test_cli:main",
            "beautyai-benchmark=beautyai_inference.cli.benchmark_cli:main",
            "beautyai-models=beautyai_inference.cli.model_manager_cli:main",
            "beautyai-manage=beautyai_inference.cli.model_management_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
