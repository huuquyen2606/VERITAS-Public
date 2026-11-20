#!/bin/bash
# ==============================================================================
# MAGIC Detector - Complete Training Pipeline
# ==============================================================================
# Prerequisites:
#   source detector/det_venv/bin/activate
#
# Usage:
#   bash run_magic_pipeline.sh [step]
#   Steps: all, prepare, acfg, train, predict
#   Default: all
# ==============================================================================

set -e
export PATH=$HOME/.local/bin:$PATH

STEP="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DETECTOR_DIR="$(dirname "$SCRIPT_DIR")"
MALDEFENDER_DIR="${SCRIPT_DIR}/maldefender"
MSACFG_DIR="${DETECTOR_DIR}/MSACFG"
VENV_DIR="${DETECTOR_DIR}/det_venv"
PYTHON="${VENV_DIR}/bin/python3"

# Dataset configuration
DATA_NAME="MSACFG"  # Must match acfg_pipeline.py outputTxtName
GPU_ID=0
MODE="cpu"  # Change to "gpu" if CUDA available
NUM_WORKERS=4

# Paths
TRAIN_DIR="${MSACFG_DIR}/TrainSet"
TEST_DIR="${MSACFG_DIR}/TestSet"
LABEL_PATH="${MSACFG_DIR}/trainLabels.csv"
HP_PATH="${MALDEFENDER_DIR}/custom.hp"

echo "============================================================"
echo " MAGIC Detector Training Pipeline"
echo "============================================================"
echo " Step:         ${STEP}"
echo " Python:       ${PYTHON}"
echo " Data dir:     ${MSACFG_DIR}"
echo " Mode:         ${MODE}"
echo "============================================================"

# Verify venv exists
if [ ! -f "${PYTHON}" ]; then
    echo "[ERROR] Virtual environment not found at ${VENV_DIR}"
    echo "Run: python3 -m venv ${VENV_DIR}"
    exit 1
fi

# ---------------------
# Hyperparameters
# ---------------------
create_hp_file() {
    cat > "${HP_PATH}" << 'EOF'
# DGCNN HPs for custom dataset (6 classes)
poolingRatio = [0.64]
graphConvSize = [[128, 64, 32, 32]]
poolingType = ['adaptive']
conv2dChannels = [16]

# Used ONLY if pooling == 'sort'
remLayers = ['conv1d']
s2vOutDim = [32]
convChannels = [[16, 32]]
convKernSizes = [[0, 5]]
convMaxPool = [[2, 2]]

# MLP
mlpType = ['vanilla']
mlpHidden = [512]
dropOutRate = [0.1]

# Training HPs
lr = [0.0001]
batchSize = [10]
cvFold = [5]
numEpochs = [30]
l2RegFactor = [0.0001]

noteMsg = ["Custom 6-class malware dataset"]
EOF
    echo "  Hyperparameters written to ${HP_PATH}"
}

# ---------------------
# Step 0: Build S2V
# ---------------------
build_s2v() {
    echo ""
    echo "[Step 0] Building Structure2Vec C++ library..."
    S2V_DIR="${MALDEFENDER_DIR}/pytorch_structure2vec-master/s2v_lib"
    if [ ! -f "${S2V_DIR}/build/dll/libs2v.so" ]; then
        make -C "${S2V_DIR}" clean 2>/dev/null || true
        make -C "${S2V_DIR}" -j$(nproc)
        echo "  S2V library built successfully."
    else
        echo "  S2V library already built. Skipping."
    fi
}

# ---------------------
# Step 1: Convert .exe -> .asm
# ---------------------
prepare_data() {
    echo ""
    echo "[Step 1] Converting .exe files to .asm format..."
    
    TRAIN_ASM_COUNT=$(ls "${TRAIN_DIR}"/*.asm 2>/dev/null | wc -l)
    if [ "${TRAIN_ASM_COUNT}" -gt 0 ]; then
        echo "  Found ${TRAIN_ASM_COUNT} .asm files in TrainSet. Skipping."
        echo "  (Delete ${TRAIN_DIR} to re-convert)"
    else
        ${PYTHON} "${MSACFG_DIR}/prepare_data.py" \
            --workers ${NUM_WORKERS} \
            --train-output "${TRAIN_DIR}" \
            --test-output "${TEST_DIR}"
    fi
}

# ---------------------
# Step 2: Build ACFG
# ---------------------
build_acfg() {
    echo ""
    echo "[Step 2] Building Attributed Control Flow Graphs..."
    
    cd "${MALDEFENDER_DIR}"
    
    # Check if ACFG already built
    if [ -f "${TRAIN_DIR}/${DATA_NAME}.txt" ]; then
        echo "  TrainSet ACFG already exists. Skipping."
    else
        echo "  Building ACFG for TrainSet..."
        ${PYTHON} acfg_pipeline.py -object train
    fi
    
    if [ -f "${TEST_DIR}/${DATA_NAME}.txt" ]; then
        echo "  TestSet ACFG already exists. Skipping."
    else
        echo "  Building ACFG for TestSet..."
        ${PYTHON} acfg_pipeline.py -object test
    fi
}

# ---------------------
# Step 3: Train Model
# ---------------------
train_model() {
    echo ""
    echo "[Step 3] Training MAGIC model..."
    
    create_hp_file
    mkdir -p "${MALDEFENDER_DIR}/${DATA_NAME,,}_models"
    
    cd "${MALDEFENDER_DIR}"
    ${PYTHON} pretrain_model.py \
        -seed 1 \
        -data ${DATA_NAME} \
        -train_dir "${TRAIN_DIR}" \
        -mode ${MODE} \
        -gpu_id ${GPU_ID} \
        -use_cached_data False \
        -cache_path "cached_${DATA_NAME,,}_graphs" \
        -norm_path "norm_${DATA_NAME,,}" \
        -norm_op min_max \
        -hp_path "${HP_PATH}"
    
    echo "  Model saved in ${MALDEFENDER_DIR}/${DATA_NAME,,}_models/"
}

# ---------------------
# Step 4: Predict on TestSet
# ---------------------
predict_model() {
    echo ""
    echo "[Step 4] Predicting on TestSet..."
    
    # Find latest model
    LATEST_MODEL=$(ls -t "${MALDEFENDER_DIR}/${DATA_NAME,,}_models/"*.pt 2>/dev/null | head -1)
    if [ -z "$LATEST_MODEL" ]; then
        echo "  [ERROR] No trained model found!"
        exit 1
    fi
    
    MODEL_DATE=$(basename "$LATEST_MODEL" | cut -d'_' -f1)
    echo "  Using model: ${LATEST_MODEL}"
    
    cd "${MALDEFENDER_DIR}"
    ${PYTHON} predict_model.py \
        -seed 1 \
        -data ${DATA_NAME} \
        -train_dir "${TRAIN_DIR}" \
        -test_dir "${TEST_DIR}" \
        -mode ${MODE} \
        -gpu_id ${GPU_ID} \
        -use_cached_data True \
        -cache_path "cached_${DATA_NAME,,}_graphs" \
        -norm_path "norm_${DATA_NAME,,}" \
        -norm_op min_max \
        -model_date "${MODEL_DATE}" \
        -hp_path "${HP_PATH}"
    
    echo "  Results: ${TEST_DIR}/submission.csv"
}

# ---------------------
# Main
# ---------------------
build_s2v

case "${STEP}" in
    all)
        prepare_data
        build_acfg
        train_model
        predict_model
        ;;
    prepare)
        prepare_data
        ;;
    acfg)
        build_acfg
        ;;
    train)
        create_hp_file
        train_model
        ;;
    predict)
        predict_model
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: bash run_magic_pipeline.sh [all|prepare|acfg|train|predict]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo " Pipeline step '${STEP}' complete!"
echo "============================================================"
