# Source this file from Bash to use this project's LeRobot installation.
_lerobot_project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_lerobot_project_dir}/.venv/bin/activate"
# TorchCodec needs the shared FFmpeg libraries installed with Miniforge.
_lerobot_ffmpeg_dir="$("${_lerobot_project_dir}/.venv/bin/python" -I -c 'import pathlib,sys,yaml; print(yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())["ffmpeg_library_dir"])' "${_lerobot_project_dir}/config/environment.yaml")" || return 1
if [[ ! -r "${_lerobot_ffmpeg_dir}/libavutil.so.60" ]]; then
    printf 'FFmpeg shared libraries missing at %s; update config/environment.yaml before running TorchCodec.\n' "${_lerobot_ffmpeg_dir}" >&2
    unset _lerobot_project_dir _lerobot_ffmpeg_dir
    return 1
fi
export LD_LIBRARY_PATH="${_lerobot_ffmpeg_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
unset _lerobot_project_dir
unset _lerobot_ffmpeg_dir
