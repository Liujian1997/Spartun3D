# #!/usr/bin/env bash
# set -euo pipefail

# # -----------------------------
# # 用户可调整的参数（如需）
# # -----------------------------
# VLLM_PORT=3000
# VLLM_CMD="CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server \
#   --model /data2/liujian/checkpoints/Qwen2.5-32B-Instruct-AWQ \
#   --host 0.0.0.0 \
#   --port ${VLLM_PORT} \
#   --gpu-memory-utilization 0.95 \
#   --max-model-len 16384 \
#   > model_info.log 2>&1 &"

# SGLANG_ENV="sglang_infer"
# LEO_ENV="leo"

# conda_activate() {
#   if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
#     source "$HOME/anaconda3/etc/profile.d/conda.sh"
#   else
#     echo "未找到 conda.sh，请确认 conda 安装路径" >&2
#     exit 1
#   fi
#   conda activate "$1"
# }

# is_vllm_running() {
#   # 1) 进程名匹配
#   if pgrep -f "vllm\.entrypoints\.openai\.api_server" >/dev/null 2>&1; then
#     return 0
#   fi
#   # 2) 端口占用匹配
#   if command -v lsof >/dev/null 2>&1; then
#     if lsof -i :${VLLM_PORT} >/dev/null 2>&1; then
#       return 0
#     fi
#   elif command -v ss >/dev/null 2>&1; then
#     if ss -ltn "( sport = :${VLLM_PORT} )" | grep -q ":${VLLM_PORT}"; then
#       return 0
#     fi
#   fi
#   return 1
# }

# if is_vllm_running; then
#   echo "[Info] 检测到 vLLM 已在运行，跳过启动。"
# else
#   echo "[Info] 未检测到 vLLM 进程，开始启动..."
#   eval "$(conda shell.bash hook)"
#   conda activate "${SGLANG_ENV}"
#   bash -lc "${VLLM_CMD}"
#   echo "[Info] 已下发启动命令：vLLM（日志：model_info.log）"
#   # 给出一点点缓冲时间（可按需调整/删除）
#   sleep 30
#   if is_vllm_running; then
#     echo "[Info] vLLM 启动成功。"
#   else
#     echo "[Warn] vLLM 似乎未成功启动，请检查 model_info.log。" >&2
#   fi
# fi

# # -----------------------------
# # 2) 激活 leo 环境并运行 5 个脚本
# # -----------------------------
eval "$(conda shell.bash hook)"
conda activate "${LEO_ENV}"

FILES=(
  /home/lj/Spartun3D/data_process/3Rscan/gpt_code/afford_can.py
  /home/lj/Spartun3D/data_process/3Rscan/gpt_code/afford.py
  /home/lj/Spartun3D/data_process/3Rscan/gpt_code/cap.py
  /home/lj/Spartun3D/data_process/3Rscan/gpt_code/object.py
  /home/lj/Spartun3D/data_process/3Rscan/gpt_code/planning.py
)

# 校验文件存在性
for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "[Error] 找不到文件：$f" >&2
    exit 1
  fi
done

# 并行运行
for f in "${FILES[@]}"; do
  base="$(basename "$f" .py)"
  nohup python "$f" > "run_${base}.log" 2>&1 &
  echo "[Run(bg)] $f -> run_${base}.log"
done

wait
echo "[Done] 全部 5 个脚本已并行结束。"