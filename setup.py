"""Setup file for dannce."""
from setuptools import setup, find_packages

setup(
    name="dannce",
    version="1.2.0",
    packages=find_packages(),
    install_requires=[
        "six",
        "pyyaml",
        "imageio==2.8.0",
        "imageio-ffmpeg",
        "numpy>1.19.0",
        "scikit-image",
        "matplotlib",
        "attr",
        "attrs",
        "multiprocess",
        "opencv-python",
        "tensorflow==2.4.0",
        "torch",
    ],
    scripts=[
        "cluster/holy_dannce_train.sh",
        "cluster/holy_dannce_train_grid.sh",
        "cluster/holy_com_train.sh",
        "cluster/holy_dannce_predict.sh",
        "cluster/holy_com_predict.sh",
        "cluster/holy_com_predict_multi_gpu.sh",
        "cluster/holy_dannce_predict_multi_gpu.sh",
        "cluster/com_and_dannce.sh",
        "cluster/dannce.sh",
        "cluster/com.sh",
        "cluster/com_and_dannce_multi_gpu.sh",
        "cluster/dannce_multi_gpu.sh",
        "cluster/com_multi_gpu.sh",
    ],
    entry_points={
        "console_scripts": [
            "dannce-train = dannce.cli:dannce_train_cli",
            "dannce-train-grid = cluster.grid:dannce_train_grid",
            "dannce-predict = dannce.cli:dannce_predict_cli",
            "com-train = dannce.cli:com_train_cli",
            "com-predict = dannce.cli:com_predict_cli",
            "dannce-predict-multi-gpu = cluster.multi_gpu:dannce_predict_multi_gpu",
            "com-predict-multi-gpu = cluster.multi_gpu:com_predict_multi_gpu",
            "dannce-predict-single-batch = cluster.multi_gpu:dannce_predict_single_batch",
            "dannce-train-single-batch = cluster.grid:dannce_train_single_batch",
            "com-predict-single-batch = cluster.multi_gpu:com_predict_single_batch",
            "dannce-merge = cluster.multi_gpu:dannce_merge",
            "com-merge = cluster.multi_gpu:com_merge",
        ]
    },
)
