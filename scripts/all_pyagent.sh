CUDA_VISIBLE_DEVICES=0 python run_agent.py \
    --model gpt-3.5-turbo-0125 --long_model gpt-3.5-turbo-0125 \
    --provider openai --dataset wtq --sub_sample False \
    --perturbation none --use_full_table True --norm True --disable_resort True --norm_cache True \
    --resume 0 --stop_at 1e6 --self_consistency 1 --temperature 0.8 \
    --log_dir output/wtq_agent --cache_dir cache/gpt-3.5 
