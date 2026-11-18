# Startup file for the interactive shell that src/hpc/gpu.sh opens on the GPU node (bash --rcfile).
# Loads the user's usual settings, then activates and checks the environment for this node's hardware.
[[ -f ~/.bashrc ]] && source ~/.bashrc
cd "${PROJECT_DIR:-/beegfs/home/mrashid/repos/XAI/Shap_BPT_SAM_tests}" || return
# shellcheck disable=SC2086
source src/hpc/node_env.sh ${NODE_ENV_ARGS:-}
echo "GPU session ready on $(hostname -s). Exit with: exit (ends the allocation)"
