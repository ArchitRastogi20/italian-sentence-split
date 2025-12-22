#!/bin/bash

# Run all 7 strategies for sentence splitting
# Optimized for RTX A4500

set -e

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
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null

MODE=${1:-help}

case $MODE in
    "quick")
        echo -e "\n${GREEN}Quick test - Strategy 5 on dev with llama-3.1-1b${NC}"
        python3 run_strategies.py --strategy 5 --model llama-3.1-1b --dataset dev
        ;;
    
    "single")
        STRAT=${2:-5}
        MODEL=${3:-llama-3.1-1b}
        DATASET=${4:-dev}
        echo -e "\n${GREEN}Running Strategy ${STRAT} with ${MODEL} on ${DATASET}${NC}"
        python3 run_strategies.py --strategy $STRAT --model $MODEL --dataset $DATASET
        ;;
    
    "all-local")
        echo -e "\n${GREEN}Running ALL strategies with LOCAL models on dev+ood${NC}"
        python3 run_strategies.py --all --local-only --all-datasets
        ;;
    
    "all-openrouter")
        echo -e "\n${GREEN}Running ALL strategies with OpenRouter models on dev+ood${NC}"
        python3 run_strategies.py --all --models gpt-oss-20b kimi-k2 --all-datasets
        ;;
    
    "all")
        echo -e "\n${GREEN}Running ALL strategies with ALL models on ALL datasets${NC}"
        python3 run_strategies.py --all --all-models --all-datasets
        ;;
    
    "dev")
        echo -e "\n${GREEN}Running ALL strategies on DEV dataset${NC}"
        python3 run_strategies.py --all --local-only --dataset dev
        ;;
    
    "ood")
        echo -e "\n${GREEN}Running ALL strategies on OOD dataset${NC}"
        python3 run_strategies.py --all --local-only --dataset ood
        ;;
    
    "strategy")
        STRAT=${2:-1}
        echo -e "\n${GREEN}Running Strategy ${STRAT} with all local models${NC}"
        python3 run_strategies.py --all --strategies $STRAT --local-only --all-datasets
        ;;
    
    "fast")
        echo -e "\n${GREEN}Running fast strategies (3,4,5) with local models${NC}"
        python3 run_strategies.py --all --strategies 3 4 5 --local-only --all-datasets
        ;;
    
    "help"|*)
        echo -e "\n${BLUE}Usage: ./run_all.sh [mode] [args...]${NC}"
        echo ""
        echo "Modes:"
        echo "  quick              - Quick test (strategy 5, llama-3.1-1b, dev)"
        echo "  single S M D       - Run strategy S with model M on dataset D"
        echo "  all-local          - All strategies, local models, dev+ood"
        echo "  all-openrouter     - All strategies, OpenRouter models, dev+ood"
        echo "  all                - Everything (all strategies, all models, all datasets)"
        echo "  dev                - All strategies, local models, dev only"
        echo "  ood                - All strategies, local models, ood only"
        echo "  strategy S         - Strategy S with all local models"
        echo "  fast               - Fast strategies (3,4,5) only"
        echo ""
        echo "Strategies:"
        echo "  1 - Sliding Window Binary Classification"
        echo "  2 - Next-Token Probability Analysis"
        echo "  3 - Marker Insertion"
        echo "  4 - Structured JSON Output"
        echo "  5 - Few-Shot Learning with Hard Examples"
        echo "  6 - Chain-of-Thought Reasoning"
        echo "  7 - Iterative Refinement"
        echo ""
        echo "Models:"
        echo "  Local: llama-3.1-1b, llama-3.1-3b"
        echo "  OpenRouter: gpt-oss-20b, kimi-k2"
        echo ""
        echo "Examples:"
        echo "  ./run_all.sh quick"
        echo "  ./run_all.sh single 5 llama-3.1-1b dev"
        echo "  ./run_all.sh strategy 6"
        echo "  ./run_all.sh all-local"
        exit 0
        ;;
esac

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${GREEN}Results in: ./results/{dev,ood}/{strategy_name}_{id}/${NC}"
echo -e "${GREEN}========================================${NC}"

# Show generated files
if [ -d "results" ]; then
    echo -e "\n${YELLOW}Generated files:${NC}"
    find results -name "*.csv" -type f 2>/dev/null | sort | head -50
fi
