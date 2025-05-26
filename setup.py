from setuptools import setup, find_packages

setup(
    name="beautyaiinference",
    version="1.0.0",
    description="A scalable, professional-grade CLI framework for running inference with Arabic AI models and multilingual language models",
    long_description="BeautyAI Inference Framework provides a unified CLI interface for managing and running inference with various language models, specializing in Arabic AI models but supporting multilingual capabilities. Features include modular architecture, multiple backend support (Transformers/vLLM), quantization capabilities, and comprehensive model lifecycle management.",
    author="Lumi AI",
    author_email="lumi@beautyai.dev",
    url="https://github.com/lumiai/beautyai-inference",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "psutil>=5.9.0",
        "argcomplete>=3.0.0",  # Add auto-completion support
    ],
    extras_require={
        "vllm": ["vllm>=0.2.0"],
        "dev": ["black", "isort", "flake8", "pytest"],
    },
    entry_points={
        "console_scripts": [
            # Primary unified CLI entry point
            "beautyai=beautyai_inference.cli.unified_cli:main",
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
