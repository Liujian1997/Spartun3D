import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["torch>=1.4"]

# ---- 改动 1：不要硬编码一堆已被 CUDA 12.x 移除的老架构 ----
# 旧代码（会触发 compute_37 等报错）：
# os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
#
# 新逻辑：
# - 若用户已通过环境变量显式指定，则尊重之（便于在其他机器/多卡复用）。
# - 否则默认针对 L40（Ada, SM 8.9）编译，避免不必要的 gencode。
if "TORCH_CUDA_ARCH_LIST" not in os.environ:  # ← CHANGED
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"  # ← CHANGED: L40 默认只编 8.9

# 可选：如果你想在同一 wheel 里兼容更多新卡（但编译会更慢），把上面一行改成：
# os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9"  # Ampere+A100/RTX30 + Ada(L40)

# ---- 改动 2：给 nvcc 补上 C++ 标准；CUDA 12.x 下一些仓库需要 c++14/17 ----
extra_cxx_flags = ["-O3"]                              # ← CHANGED
extra_nvcc_flags = ["-O3", "-Xfatbin", "-compress-all", "-std=c++17"]  # ← CHANGED
# 如遇到 PyTorch/编译器对 ABI 的要求，还可按需追加：
# extra_cxx_flags += ["-D_GLIBCXX_USE_CXX11_ABI=0"]    # 仅在你的环境要求时启用

# （可选）在构建期打印一下关键信息，方便排错
try:  # ← CHANGED
    import torch  # 仅用于打印 cuda 版本，非强依赖
    print("[build] torch.version.cuda =", torch.version.cuda)
    print("[build] TORCH_CUDA_ARCH_LIST =", os.environ.get("TORCH_CUDA_ARCH_LIST"))
except Exception as _e:
    print("[build] torch import failed (ok to ignore):", _e)

exec(open("_version.py").read())

setup(
    name='pointnet2',
    version=__version__,
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            extra_compile_args={                 # ← CHANGED
                "cxx": extra_cxx_flags,          # ← CHANGED
                "nvcc": extra_nvcc_flags,        # ← CHANGED
            },
            include_dirs=[osp.join(_this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)