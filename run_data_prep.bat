@echo off
setlocal
set PY=python

echo Starting Reddit Data Preparation Pipeline
echo ==================================================

%PY% scripts/00_env_check.py --config configs/exp_small.yaml || goto :eof
echo.
echo Environment check completed successfully!
echo.

%PY% scripts/00_data_preparation.py --config configs/exp_small.yaml || goto :eof
echo.
echo Data preparation with Davidson labeling completed successfully!
echo.

echo ==================================================
echo DONE - Data Preparation completed!
echo Check artifacts/ and figures/ directories for results.
echo ==================================================
endlocal
