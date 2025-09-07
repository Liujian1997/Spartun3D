import os                               # ← CHANGED
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 若外部未显式设置架构列表，则默认只为 L40 (SM 8.9) 编译
# 这样可避免 CUDA 12.x 下对老架构（compute_37 等）报错
if "TORCH_CUDA_ARCH_LIST" not in os.environ:          # ← CHANGED
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"        # ← CHANGED

# 推荐使用 O3 优化，并给 nvcc 加上 C++ 标准（很多仓库在 CUDA 12.x 下需要）
extra_cxx = ["-O3"]                                    # ← CHANGED
extra_nvcc = ["-O3", "-Xfatbin", "-compress-all", "-std=c++17"]  # ← CHANGED
# 如遇到 ABI 不匹配（视你本机 PyTorch/编译器而定），可按需开启下一行：
# extra_cxx += ["-D_GLIBCXX_USE_CXX11_ABI=0"]          # ← OPTIONAL

setup(
    name='pointnet2_cuda',
    ext_modules=[
        CUDAExtension(
            'pointnet2_batch_cuda',
            sources=[
                'src/pointnet2_api.cpp',
                'src/ball_query.cpp',
                'src/ball_query_gpu.cu',
                'src/group_points.cpp',
                'src/group_points_gpu.cu',
                'src/interpolate.cpp',
                'src/interpolate_gpu.cu',
                'src/sampling.cpp',
                'src/sampling_gpu.cu',
            ],
            extra_compile_args={
                'cxx': extra_cxx,                      # ← CHANGED
                'nvcc': extra_nvcc,                    # ← CHANGED
            },
            # 如果你的头文件在自定义目录，可在此补充 include_dirs（可选）
            # include_dirs=['include'],               # ← OPTIONAL
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)