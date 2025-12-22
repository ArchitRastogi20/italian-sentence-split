#!/bin/bash

# Run all 7 strategies for sentence splitting
# Optimized for RTX A4500

set -e

# Always run from project root
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Sentence Splitting - 7 Strategies${NC}"
echo -e "${GREEN}========================================${NC}"

# Check CUDA
echo -e "\n${YELLOW}Checking CUDA...${NC}"
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || true

MODE=${1:-help}

RUN_CMD="python3 strategies_src/run_strategies.py"

case $MODE in
    "quick")
        echo -e "\n${GREEN}Quick test - Strategy 5 on dev with llama-3.1-1b${NC}"
        $RUN_CMD --strategy 5 --model llama-3.1-1b --dataset dev
        ;;
    
    "single")
        STRAT=${2:-5}
        MODEL=${3:-llama-3.1-1b}
        DATASET=${4:-dev}
        echo -e "\n${GREEN}Running Strategy ${STRAT} with ${MODEL} on ${DATASET}${NC}"
        $RUN_CMD --strategy "$STRAT" --model "$MODEL" --dataset "$DATASET"
        ;;
    
    "all-local")
        echo -e "\n${GREEN}Running ALL strategies with LOCAL models on dev+ood${NC}"
        $RUN_CMD --all --local-only --all-datasets
        ;;
    
    "all-openrouter")
        echo -e "\n${GREEN}Running ALL strategies with OpenRouter models on dev+ood${NC}"
        $RUN_CMD --all --models gpt-oss-20b kimi-k2 --all-datasets
        ;;
    
    "all")
        echo -e "\n${GREEN}Running ALL strategies with ALL models on ALL datasets${NC}"
        $RUN_CMD --all --all-models --all-datasets
        ;;
    
    "dev")
        echo -e "\n${GREEN}Running ALL strategies on DEV dataset${NC}"
        $RUN_CMD --all --local-only --dataset dev
        ;;
    
    "ood")
        echo -e "\n${GREEN}Running ALL strategies on OOD dataset${NC}"
        $RUN_CMD --all --local-only --dataset ood
        ;;
    
    "strategy")
        STRAT=${2:-1}
        echo -e "\n${GREEN}Running Strategy ${STRAT} with all local models${NC}"
        $RUN_CMD --strategies "$STRAT" --local-only --all-datasets
        ;;
    
    "fast")
        echo -e "\n${GREEN}Running fast strategies (3,4,5) with local models${NC}"
        $RUN_CMD --strategies 3 4 5 --local-only --all-datasets
        ;;
    
    "help"|*)
        echo -e "\n${BLUE}Usage: ./run_all.sh [mode] [args...]${NC}"
        echo ""
        echo "Modes:"
        echo "  quick              - Quick test (strategy 5, llama-3.1-1b, dev)"
        echo "  single S M D       - Run strategy S with model M on dataset D"
        echo "  all-local          - All strategies, local models, dev+ood"
        echo "  all-openrouter     - All strategies, OpenRouter models, dev+ood"
        echo "  all                - Everything"
        echo "  dev                - All strategies, local models, dev only"
        echo "  ood                - All strategies, local models, ood only"
        echo "  strategy S         - Single strategy with all local models"
        echo "  fast               - Fast strategies (3,4,5)"
        echo ""
        exit 0
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}Results in: ./results/{dev,ood}/${NC}"
echo -e "${GREEN}========================================${NC}"

# Show generated files
if [ -d "results" ]; then
    echo -e "\n${YELLOW}Generated files:${NC}"
    find results -name "*.csv" -type f 2>/dev/null | sort | head -50
fi
