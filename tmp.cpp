@echo off
setlocal

cd /d C:\your_project

set sizes=4 8 16 32 64 128

for %%h in (%sizes%) do (
    for %%w in (%sizes%) do (
        echo [RUN] h=%%h w=%%w
        python run_experiment.py --h %%h --w %%w
    )
)

echo All jobs finished
pause
