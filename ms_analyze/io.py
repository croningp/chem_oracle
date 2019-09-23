import subprocess


def datx_to_npz(filename: str, python_executable: str):
    subprocess.run(
        [python_executable, "-m", "advion_wrapper.AdvionData", f'"{filename}"']
    )
