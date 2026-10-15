#%%

import subprocess

def hydrofon(wlacz):
    subprocess.run(
        ["amixer", "-q", "-c", "1", "sset", "Direct Monitor",
         "Mono" if wlacz else "Off"],
        check=True,
        timeout=5,
    )


if __name__ == "__main__":
    hydrofon(True)   # włącz
    hydrofon(False)  # wyłącz
