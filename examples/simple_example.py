from diffulex import Diffulex, SamplingParams
from transformers import AutoTokenizer

if __name__ == '__main__':
    # Initialize the Diffulex engine
    model_path = "/home/hoffmuki/scratch/models/Fast_dLLM_v2_7B"
    llm = Diffulex(
        model_path,
        model_name="fast_dllm_v2",  # or "dream", "llada", etc.
        tensor_parallel_size=2,
        data_parallel_size=1,
        gpu_memory_utilization=0.5,
        max_model_len=2048,
        decoding_strategy="fast_dllm",  # or "d2f", "fast_dllm"
        mask_token_id=151665,  # model-specific mask token ID
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=256,
    )

    # Prepare prompts
    prompts = [
        "Question: What is the capital of France? Answer:",
        "Question: Explain quantum computing in simple terms. Answer:",
    ]

    # Generate responses
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    for output in outputs:
        print(f"Generated text: {output['text']}")
        print(f"Number of diffusion steps: {output['n_diff_steps']}")
        print(f"Token IDs: {output['token_ids']}")