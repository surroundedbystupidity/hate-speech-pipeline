@echo off
setlocal
set PY=python

echo =========================================================
echo Reddit Hate Speech Analysis - Complete Research Pipeline
echo =========================================================
echo.

echo [1/8] Environment Check...
%PY% scripts/00_env_check.py --config configs/exp_small.yaml || goto :error
echo  Environment check completed
echo.

echo [2/8] Data Preparation with Davidson Labeling...
%PY% scripts/00_data_preparation.py --config configs/exp_small.yaml || goto :error
echo  Data preparation completed
echo.

echo [3/8] BERT Feature Extraction...
%PY% scripts/01_bert_feature_extraction.py --config configs/exp_small.yaml || goto :error
echo BERT feature extraction completed
echo.

echo [4/8] Temporal Graph Construction...
%PY% scripts/02_temporal_graph_construction.py --config configs/exp_small.yaml || goto :error
echo Temporal graph construction completed
echo.

echo [5/8] TGNN Model Training...
%PY% scripts/03_tgnn_model.py --config configs/exp_small.yaml || goto :error
echo TGNN model training completed
echo.

echo [6/8] Diffusion Prediction Analysis...
%PY% scripts/04_diffusion_prediction.py --config configs/exp_small.yaml || goto :error
echo Diffusion prediction completed
echo.

echo [7/8] Moderation Strategy Simulation...
%PY% scripts/05_moderation_simulation.py --config configs/exp_small.yaml || goto :error
echo  Moderation simulation completed
echo.

echo [8/8] Advanced Evaluation and Visualization...
%PY% scripts/06_advanced_evaluation.py --config configs/exp_small.yaml || goto :error
%PY% scripts/07_research_visualization.py --config configs/exp_small.yaml || goto :error
echo  Advanced evaluation and visualization completed
echo.

echo =========================================================
echo COMPLETE PIPELINE FINISHED SUCCESSFULLY!
echo =========================================================
echo.
echo Results saved to:
echo   artifacts/ - All analysis results and trained models
echo   figures/   - Visualizations and research plots
echo.
echo Key outputs:
echo   - Balanced hate speech dataset
echo   - BERT embeddings and user features
echo   - Temporal graph structure
echo   - Trained TGNN models (TGAT/TGN)
echo   - Diffusion prediction results
echo   - Moderation strategy analysis
echo   - Comprehensive evaluation report
echo   - Publication-ready visualizations
echo.
echo Check comprehensive_evaluation_report.json for detailed results!
echo =========================================================
goto :end

:error
echo.
echo ERROR: Pipeline failed at step %errorlevel%
echo Please check the error message above and fix the issue.
echo.
pause
exit /b 1

:end
endlocal
pause
