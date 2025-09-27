@echo off
echo ========================================
echo Reddit Hate Speech Analysis - GPU 7 Days
echo ========================================
echo.
echo 配置: GPU优化，7天数据，大规模处理
echo 开始时间: %date% %time%
echo.

echo [1/8] 环境检查...
python scripts/00_env_check.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 环境检查失败！
    pause
    exit /b 1
)

echo.
echo [2/8] 数据准备 (7天数据)...
python scripts/00_data_preparation.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 数据准备失败！
    pause
    exit /b 1
)

echo.
echo [3/8] BERT特征提取 (GPU加速)...
python scripts/01_bert_feature_extraction.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo BERT特征提取失败！
    pause
    exit /b 1
)

echo.
echo [4/8] 时序图构建...
python scripts/02_temporal_graph_construction.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 时序图构建失败！
    pause
    exit /b 1
)

echo.
echo [5/8] TGNN模型训练 (GPU加速)...
python scripts/03_tgnn_model.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo TGNN模型训练失败！
    pause
    exit /b 1
)

echo.
echo [6/8] 扩散预测...
python scripts/04_diffusion_prediction.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 扩散预测失败！
    pause
    exit /b 1
)

echo.
echo [7/8] 内容审核模拟...
python scripts/05_moderation_simulation.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 内容审核模拟失败！
    pause
    exit /b 1
)

echo.
echo [8/8] 高级评估和可视化...
python scripts/06_advanced_evaluation.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 高级评估失败！
    pause
    exit /b 1
)

python scripts/07_research_visualization.py --config configs/exp_gpu_7days.yaml
if %errorlevel% neq 0 (
    echo 研究可视化失败！
    pause
    exit /b 1
)

echo.
echo ========================================
echo GPU 7天数据分析完成！
echo 结束时间: %date% %time%
echo ========================================
echo.
echo 结果文件位置:
echo - 模型文件: artifacts/
echo - 可视化: figures/
echo - 评估报告: artifacts/comprehensive_evaluation_report.json
echo.
pause
