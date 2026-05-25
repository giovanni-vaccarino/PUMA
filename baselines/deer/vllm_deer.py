import warnings
warnings.filterwarnings("ignore")  # Ignore all warnings

import os
import json
import time
import argparse
import sys
# Add project root to sys.path so `baselines.*` imports work when run as a script
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import torch
import torch.nn.functional as F
from vllm.outputs import CompletionOutput
from typing import Any, Dict, List
from nltk import ngrams
from collections import Counter

from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams
from baselines.utils.generation_config import load_generation_params
from baselines.utils.math_util import extract_code_answer, CODE_SYS_PROMPT, CODE_ANSWER_SUFFIX
import pdb

import math
import numpy as np
import random

def set_seeds(seed=42):
    # Set Python built-in random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch CPU random seed
    torch.manual_seed(seed)

    # If using GPU (especially CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)           # Set seed for current GPU
        torch.cuda.manual_seed_all(seed)       # Also effective for multi-GPU

        # For better reproducibility, enable cudnn determinism mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Optional: Set generator (for DataLoader with multi-threading)
    g = torch.Generator()
    g.manual_seed(seed)
    




def append_jsonl(data, file_path):
    """Append results in the list to a .jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def write_jsonl(data, file_path):
    """Write results in the list to a .jsonl file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def read_jsonl(file_path):
    """Read .jsonl file and return a list of dictionaries."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def seq_rep_n(last_thinking, cur_thinking, rep, n=1):

    
    pred = last_thinking
    target = cur_thinking

    pred_tokens = pred.split(' ')
    target_token = target.split(' ')

    ngs_pred = [ng for ng in ngrams(pred_tokens, n)]
    ngs_know = [ng for ng in ngrams(target_token, n)]
    intersection = list(set(ngs_pred) & set(ngs_know))
    overlap_num = len(intersection)
    
    
    if overlap_num == len(ngs_pred) and overlap_num == len(ngs_know):
        rep += 1
        
    
    return rep

# Function to calculate average max probability, mimicking Transformers version logic
def calculate_average_max_prob_from_logprobs(logprobs_list, policy='avg2') -> float:
    """
    Calculate average max token probability from logprobs list in vLLM CompletionOutput.
    Compute from the second generated token to the second-to-last token.
    policy: min, avg1: arithmetic mean, avg2: geometric mean
    """

    num_tokens = len(logprobs_list)
    start_index = 1
    end_index = num_tokens

    if num_tokens < 1:
        print("Too few tokens to calculate valid average.")
        return 0.0

    total_prob_sum = 0.0
    log_prob_sum = 0.0  # For geometric mean
    count_for_average = 0
    min_prob = 1.0
    
    for i in range(start_index, end_index):
        # Ensure index is valid and corresponding logprobs entry is not empty
        if i < len(logprobs_list) and logprobs_list[i]:
             try:
                 logprob_obj = list(logprobs_list[i].values())[0]
                 # Ensure object has .logprob attribute
                 if hasattr(logprob_obj, 'logprob'):
                     prob = torch.exp(torch.tensor(logprob_obj.logprob)).item()
                     if prob < min_prob:
                        min_prob = prob
                     #print(prob)
                     #print(list(logprobs_list[i].values())[0])
                     total_prob_sum += prob
                     log_prob_sum += math.log(max(prob, 1e-10)) 
                     count_for_average += 1
                 else:
                      print(f"Warning: Object at logprobs_list[{i}] doesn't have '.logprob' attribute.")
             except (IndexError, KeyError, AttributeError) as e:
                 print(f"Warning: Unable to process logprobs at logprobs_list[{i}]: {e}")
        else:
             print(f"Warning: logprobs_list[{i}] is empty or invalid.")
    # Calculate average
    if policy == 'min':
        result = min_prob
    elif policy == 'avg1':
        result = total_prob_sum / count_for_average
    elif policy == 'avg2':
        result = math.exp(log_prob_sum / count_for_average) 

    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./DeepSeek-R1-Distill-Qwen-14B/")
    parser.add_argument('--dataset_dir', type=str, default="../data/")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max-model-len", "--model-context-len", type=int, default=40000, dest="model_context_len") # max-model-len for vllm, should be longer than max_generated_tokens.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--run_time", type=int, default=1)
    parser.add_argument("--no_thinking", type=int, default=0) # Calculate the answer confidence at the very beginning of the reasoning process and attempt to exit early.
    parser.add_argument("--rep", type=int, default=0) # Exit early when repetition occurs, but it remains to be implemented. (TODO)
    parser.add_argument("--points", type=int, default=1) # 1: 'Wait' as thinking transition point. 0: 'Alternatively' as thinking transition point. 
    parser.add_argument("--af", type=int, default=0) # answer forcing at end of sequence
    parser.add_argument("--max_judge_steps", type=int, default=10) # Limit the maximum number of answer attempts to save time cost.
    parser.add_argument('--policy', type=str, default="avg1") # Strategy for Calculating Answer Confidence

    parser.add_argument('--threshold', type=float, default=0.95) # The answer confidence threshold used to determine early exit.
    parser.add_argument('--max_generated_tokens', '--max-len', type=int, default=32768, dest="max_len") # total token budget
    parser.add_argument('--dataset', type=str, default='aime24') # dataset name
    parser.add_argument('--output_path', type=str, default='./outputs/deer.jsonl') # output path # ./outputs/deer_vanilla.jsonl
    parser.add_argument('--think_ratio', type=float, default=0.9, help="Ratio of thinking phase to max generated tokens") # Ratio of thinking phase to max generated tokens
    parser.add_argument('--batch_size', type=int, default=2000) # vllm batch size, set it to a value above the number of samples in the dataset.
    parser.add_argument('--seed', type=int, default=0, help="Random seed for vLLM (default: 0)")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top_p', type=float, default=0.95)
    
    # Hardcoded 20
    parser.add_argument('--prob_check_max_tokens', type=int, default=20, help="Max tokens for probability check phase") # Max tokens for answer inducing
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load generation params from model's generation_config.json
    gen_params = load_generation_params(args.model_name_or_path)
    args.temperature = gen_params["temperature"]
    args.top_p = gen_params["top_p"]
    args.top_k = gen_params["top_k"]

    args.model_context_len = args.max_len + 8000
    print(f"Using vLLM LLM object for direct inference (batch processing)")
    print(f"Model path: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Early exit probability threshold: {args.threshold}")
    print(f"Max total generated tokens: {args.max_len}")
    print(f"Thinking phase ratio: {args.think_ratio}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max tokens for probability check phase: {args.prob_check_max_tokens}")

    print("\nInitializing vLLM LLM engine...")
    available_gpus = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')

    try:
        llm_engine = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus),
            dtype=args.dtype,
            max_model_len=args.max_len + 8000,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            seed=args.seed,
        )
        print("vLLM LLM engine initialized successfully.")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        print(f"Successfully loaded tokenizer: {args.model_name_or_path}")
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                 tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                 print("Warning: Model has no pad_token or eos_token. Added custom [PAD] token.")

            print(f"Tokenizer using pad_token_id: {tokenizer.pad_token_id}")


    except Exception as e:
        print(f"Failed to initialize vLLM LLM engine or load tokenizer: {e}")
        sys.exit(1)

    is_code = "livecodebench" in args.dataset.lower()
    if is_code:
        sys_prompt = CODE_SYS_PROMPT
        args.prob_check_max_tokens = 200  # code answers are longer
        print(f"Code task detected: using code system prompt, prob_check_max_tokens={args.prob_check_max_tokens}")
    else:
        sys_prompt = 'Please reason step by step, and put your final answer within \\boxed{}.'
    dataset_path = f'{args.dataset_dir}/{args.dataset}_test.jsonl'
    try:
        questions_json = read_jsonl(dataset_path)
        if not questions_json:
            print(f"Error: No questions loaded from {dataset_path}.")
            sys.exit(1)
        print(f"Successfully loaded dataset: {dataset_path}, total {len(questions_json)} questions")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        sys.exit(1)

    output_file = args.output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"\nStarting processing, total questions: {len(questions_json)}")
    start_time = time.time()

    questions_state = {} # Dictionary to store processing state for each question
    last_token_strs = ["</think>"] # Strings marking end of thinking
    if args.points == 1:
        continue_str = "Wait" # String appended to sequence end to indicate continued thinking
    else:
        continue_str = "Alternatively" # String appended to sequence end to indicate continued thinking

    if is_code:
        answer_prompt_str = CODE_ANSWER_SUFFIX  # "\n### Solution Code\n```python\n"
    elif 'gpqa' in args.dataset:
        answer_prompt_str = "\n**Final Answer**\nI believe the final answer, rather than the option, is \\boxed"
    else:
        answer_prompt_str = "\n**Final Answer**\n\\boxed" # Prompt string to guide answer generation

    # Get token IDs for stop conditions and strings to append
    last_token_ids = []
    for s in last_token_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: last_token_ids.extend(ids)
    last_token_ids = list(set(last_token_ids)) # Remove duplicate IDs

    continue_ids = tokenizer.encode(continue_str, add_special_tokens=False)
    if not continue_ids:
         print(f"Warning: Unable to tokenize continue string '{continue_str}'. This may affect logic.")

    # Stop tokens for thinking phase generation
    generation_stop_tokens = [continue_str] + last_token_strs + [tokenizer.eos_token]
    if is_code:
        pred_prob_stop_tokens = ["```\n", "```"]  # end of code block
    else:
        pred_prob_stop_tokens = [' }', '}\n', '}\n\n', '}.', '}.\n', '}\\', '}}', ')}', ')}.', ')}\n'] # where \boxed{} ends. Used to stop the model from predicting intermediate answers.

    answer_stop_tokens = [tokenizer.eos_token]
    

    # Max token limit for thinking phase
    think_limit_tokens = int(args.max_len * args.think_ratio)

    for i, question_data in enumerate(questions_json):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question_data['question']}
        ]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if args.no_thinking == 1:
            questions_state[i] = {
            'question_data': question_data,
            'state': 'needs_prob_check',  # try to exit ai begin
            'formatted_prompt': formatted_prompt, 
            'current_full_sequence': formatted_prompt, 
            'generated_thinking_history': "", 
            'generated_thinking_last_trun': "nsss, ssssa, wrtt, yyy, sss", 
            'generated_answer_history': "", 
            'pred_prob': 0.0, 
            'too_long': 0,
            'rep_end': 0,
            'high_prob': 0,
            'regular_end': 0,
            'thinking_steps': 0,
            'num_trial_answer_tokens': 0,
            'output_dict': {}, 
            'error_message': None, 
            'question_index': i 
        }
        else:
            questions_state[i] = {
            'question_data': question_data,
            'state': 'needs_thought_chunk', 
            'formatted_prompt': formatted_prompt, 
            'current_full_sequence': formatted_prompt, 
            'generated_thinking_history': "", 
            'generated_thinking_last_trun': "nsss, ssssa, wrtt, yyy, sss", 
            'generated_answer_history': "", 
            'pred_prob': 0.0, 
            'too_long': 0,
            'rep_end': 0,
            'high_prob': 0,
            'regular_end': 0,
            'thinking_steps': 0,
            'num_trial_answer_tokens': 0,
            'output_dict': {}, 
            'error_message': None, 
            'question_index': i 
        }

    active_questions_indices = list(questions_state.keys()) # List of currently processing question indices
    pbar = tqdm(total=len(questions_json), desc="Processing questions")

    print("\nRunning a simple test generation...")

    # Main processing loop: continue while there are active questions
    while active_questions_indices: #  indexes [0,1,2,...,n]
        batch_prompts = [] # Current batch prompts
        batch_sampling_params = [] # Current batch sampling parameters
        batch_request_info = [] # Store (question index, step type) for output processing

        current_batch_count = 0
        # Create copy of active_questions_indices to allow modifying original list during iteration
        current_active_indices_for_batching = active_questions_indices[:]

        # Build current batch
        for q_idx in current_active_indices_for_batching:
            if current_batch_count >= args.batch_size:
                 break

            state = questions_state[q_idx]
            if state['state'] in ['finished', 'error']:
                 continue

            prompt_for_batch = None
            sampling_params_for_batch = None
            step_type = None # 'think', 'prob_check_gen', 'answer'

            try:
                # --- Determine prompt and parameters based on state ---
                current_full_sequence_tokens = tokenizer.encode(state['current_full_sequence'], add_special_tokens=False)
                current_full_sequence_len = len(current_full_sequence_tokens)
                current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
                initial_prompt_len = len(tokenizer.encode(state['formatted_prompt'], add_special_tokens=False))

                # Check context window limit before adding any new content
                remaining_context_window = args.model_context_len - current_full_sequence_len
                # check window, skip it
                if remaining_context_window <= 0:
                     state['state'] = 'error'
                     state['error_message'] = f"Exceeded model context window limit ({args.model_context_len}). Current sequence length: {current_full_sequence_len}"
                     state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message'], 'gold_answer': state['question_data']['answer']}
                     print(f"\nQuestion {q_idx}: {state['error_message']}")
                     if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)
                     continue 

                # Initial state        
                if state['state'] == 'needs_thought_chunk':
                    # Calculate max tokens for this generation chunk, considering thinking limit and context window
                    max_new_tokens_for_thought = min(
                        think_limit_tokens - current_generated_thinking_tokens, # thinking budget
                        remaining_context_window 
                    )
                    if max_new_tokens_for_thought <= 0:
                        state['state'] = 'needs_answer'
                        print(f"\nQuestion {q_idx}: Reached thinking limit ({current_generated_thinking_tokens}/{think_limit_tokens}). Switching to answer generation.")
                        state['too_long'] = 1
                        continue 
                         


                    prompt_for_batch = state['current_full_sequence']
                    if state['thinking_steps'] < args.max_judge_steps:
                        sampling_params_for_batch = SamplingParams(
                        max_tokens=max_new_tokens_for_thought,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        stop=generation_stop_tokens
                    )
                    else:
                        sampling_params_for_batch = SamplingParams(
                        max_tokens=max_new_tokens_for_thought,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        stop=last_token_strs
                    )
                    step_type = 'think'

                # Check state
                elif state['state'] == 'needs_prob_check':
                    # Build prompt for probability check generation:
                    prompt_for_prob_check = state['current_full_sequence'] + answer_prompt_str
                    prob_check_prompt_len = len(tokenizer.encode(prompt_for_prob_check, add_special_tokens=False))
                    required_space_for_prob_check = prob_check_prompt_len + args.prob_check_max_tokens

                    if required_space_for_prob_check > args.model_context_len:
                         state['state'] = 'error'
                         state['error_message'] = f"Probability check generation prompt exceeds context window ({args.model_context_len}). Estimated length: {required_space_for_prob_check}"
                         state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message'], 'gold_answer': state['question_data']['answer']}
                         print(f"\nQuestion {q_idx}: {state['error_message']}")
                         if q_idx in active_questions_indices:
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                         continue


                    # Parameters for generating *prediction sequence* in probability check phase
                    prompt_for_batch = prompt_for_prob_check
                    sampling_params_for_batch = SamplingParams(
                        max_tokens=args.prob_check_max_tokens,
                        temperature=args.temperature, 
                        stop=pred_prob_stop_tokens, # Only predict content inside \boxed{}
                        logprobs=1,
                    )
                    step_type = 'prob_check_gen' 


                elif state['state'] == 'needs_answer':
                    # Build final answer prompt
                    
                    if state['too_long'] == 1:
                        final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + '\n</think>\n\n'
                        state['generated_thinking_history'] = state['generated_thinking_history'] + '\n</think>\n\n'
                        
                    else:
                        final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + '\n</think>\n\n'#+ answer_prompt_str
                        state['generated_thinking_history'] = state['generated_thinking_history'] + '\n</think>\n\n'
                    
                    len_final_answer_prompt = len(tokenizer.encode(final_answer_prompt, add_special_tokens=False))
                    total_tokens_before_answer_prompt = current_generated_thinking_tokens
                    # Calculate remaining total budget
                    remaining_total_budget = args.max_len - total_tokens_before_answer_prompt
                    max_new_tokens_answer = min(
                       remaining_total_budget,
                       args.model_context_len - len_final_answer_prompt
                    )
                    
                    if max_new_tokens_answer <= 0:
                        
                        state['state'] = 'error'
                        state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + "\nSkipped answer generation due to length limit.", 'gold_answer': state['question_data']['answer'], 'too_long': state['too_long'], 'thinking_steps': state['thinking_steps']}
                        print(f"\nQuestion {q_idx}: Skipped answer generation due to length limit.")
                        
                        if q_idx in active_questions_indices:
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                        continue
                    else:
                         prompt_for_batch = final_answer_prompt
                         sampling_params_for_batch = SamplingParams(
                            max_tokens=max_new_tokens_answer,
                            temperature=args.temperature,
                            stop=answer_stop_tokens,
                            top_p=args.top_p,
                         )
                         step_type = 'answer'





                elif state['state'] == 'answer_forcing':
                    final_answer_prompt = state['formatted_prompt'] +  state['generated_thinking_history'] + state['generated_answer_history'] + answer_prompt_str
                    state['generated_thinking_history'] = state['generated_thinking_history'] + state['generated_answer_history'] + answer_prompt_str
                    
                    prompt_for_batch = final_answer_prompt
                    sampling_params_for_batch = SamplingParams(
                            max_tokens=100,
                            temperature=args.temperature,
                            stop=answer_stop_tokens,
                            top_p=args.top_p,
                         )
                    step_type = 'answer_exit'

                # If execution reaches here and prompt_for_batch is None, there may be state logic issues
                # This check generally shouldn't trigger
                if prompt_for_batch is None:
                     state['state'] = 'error'
                     state['error_message'] = f"Internal error: No prompt generated for state {state['state']}."
                     state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message'], 'gold_answer': state['question_data']['answer']}
                     print(f"\nQuestion {q_idx}: {state['error_message']}")
                     if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)
                     continue 
                
                batch_prompts.append(prompt_for_batch)
                batch_sampling_params.append(sampling_params_for_batch)
                batch_request_info.append((q_idx, step_type))
                current_batch_count += 1

            except Exception as e:
                state['state'] = 'error'
                state['error_message'] = f"Error preparing batch request for state '{state['state']}': {e}"
                state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message'], 'gold_answer': state['question_data']['answer']}
                print(f"\nQuestion {q_idx}: {state['error_message']}")
                if q_idx in active_questions_indices:
                    active_questions_indices.remove(q_idx)
                    pbar.update(1)

        if not batch_prompts:
            # May occur when all remaining active questions have switched to finished/error state or were skipped.
            all_stuck = True
            for q_idx in active_questions_indices:
                 if questions_state[q_idx]['state'] not in ['finished', 'error']:
                      all_stuck = False
                      break
            if all_stuck:
                 print("Warning: Batch generated no requests. All remaining questions are completed or in error state.")
                 break
            else:
                 print("Error: Active questions remain but no batch requests generated. Possible logic error.")
                 for q_idx in list(active_questions_indices):
                     state = questions_state[q_idx]
                     if state['state'] not in ['finished', 'error']:
                        state['state'] = 'error'
                        state['error_message'] = "Processing aborted: Unable to generate request in batch loop."
                        state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\n" + state['error_message'], 'gold_answer': state['question_data']['answer']}
                        print(f"\nQuestion {q_idx}: Marked as error due to processing abort.")
                        if q_idx in active_questions_indices: # Should be, but double-checking
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)
                 break

        batch_outputs = llm_engine.generate(batch_prompts, batch_sampling_params, use_tqdm=False)
    

        # --- Process batch outputs ---
        for i, output in enumerate(batch_outputs):
            q_idx, step_type = batch_request_info[i]
            state = questions_state[q_idx]

            if state['state'] in ['finished', 'error']:
                 continue

            try:
                if not output.outputs: # skip, exception handling
                    # vLLM returned empty output, possibly due to prompt issues, length limits or other internal errors
                    error_msg = f"vLLM returned empty output for request {output.request_id} (question {q_idx}, step {step_type})."
                    if hasattr(output, 'error') and output.error:
                        error_msg += f" vLLM error: {output.error}"
                    current_full_sequence_len = len(tokenizer.encode(state['current_full_sequence'], add_special_tokens=False))
                    initial_prompt_len = len(tokenizer.encode(state['formatted_prompt'], add_special_tokens=False))
                    current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
 

                    if step_type == 'think':
                         max_new = min(think_limit_tokens - current_generated_thinking_tokens, args.model_context_len - current_full_sequence_len)
                         error_msg += f" State: think, attempting to generate {max_new} tokens."
                         if max_new <= 0: error_msg += " Note: Calculated max new tokens <= 0."
                         if args.model_context_len - current_full_sequence_len <= 0: error_msg += " Note: Already exceeded context window."
                         if think_limit_tokens - current_generated_thinking_tokens <= 0: error_msg += " Note: Reached thinking token limit."

                    elif step_type == 'prob_check_gen':
                         prob_check_prompt = state['current_full_sequence'] + answer_prompt_str
                         prob_check_prompt_len = len(tokenizer.encode(prob_check_prompt, add_special_tokens=False))
                         error_msg += f" State: prob_check_gen, prompt length: {prob_check_prompt_len}, attempting to generate {args.prob_check_max_tokens} tokens."
                         if prob_check_prompt_len + args.prob_check_max_tokens > args.model_context_len: error_msg += " Note: Prompt+generation exceeds context window limit."

                    elif step_type == 'answer':
                         final_answer_prompt = state['formatted_prompt'] + state['generated_thinking_history'] + answer_prompt_str
                         len_final_answer_prompt = len(tokenizer.encode(final_answer_prompt, add_special_tokens=False))
                         total_tokens_before_answer_prompt = initial_prompt_len + current_generated_thinking_tokens
                         max_new_answer = min(args.max_len - total_tokens_before_answer_prompt, args.model_context_len - len_final_answer_prompt)
                         error_msg += f" State: answer, prompt length: {len_final_answer_prompt}, attempting to generate {max_new_answer} tokens."
                         if max_new_answer <= 0: error_msg += " Note: Calculated max new tokens <= 0."
                         if args.model_context_len - len_final_answer_prompt <= 0: error_msg += " Note: Already exceeded context window."
                         if args.max_len - total_tokens_before_answer_prompt <= 0: error_msg += " Note: Reached total token limit."

                    raise ValueError(error_msg) 

                completion_output = output.outputs[0]
                generated_text = completion_output.text
                generated_ids = completion_output.token_ids
                last_token_id = generated_ids[-1]

                rep = seq_rep_n(state['generated_thinking_last_trun'], generated_text, state['rep_end'])
                state['rep_end'] = rep
                
                
                if step_type == 'think':
                    if state['rep_end'] >= 3 and args.rep == 1: 
                        state['state'] = 'needs_answer'
                        state['generated_thinking_history'] += generated_text
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history']
                        #state['rep_end'] = 1
                    elif last_token_id in last_token_ids:
                        state['state'] = 'needs_answer'
                        state['generated_thinking_history'] += generated_text
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history']
                        state['regular_end'] = 1
                    else:
                        # Append generated thinking chunk
                        state['generated_thinking_history'] += generated_text
                        state['generated_thinking_last_trun'] = generated_text
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history']
                        state['state'] = 'needs_prob_check'
                        state['thinking_steps'] += 1

                elif step_type == 'prob_check_gen':
                    # Get logprobs for probability calculation.
                    if completion_output.logprobs:
                         state['pred_prob'] = calculate_average_max_prob_from_logprobs(completion_output.logprobs, args.policy)
                         
                    else:
                         print(f"Warning: No logprobs returned for prob_check_gen for question {q_idx}. Setting pred_prob to 0.0.")
                         state['pred_prob'] = 0.0

                    # Accumulate trial answer tokens for CRT calculation
                    state['num_trial_answer_tokens'] += len(completion_output.token_ids)

                    # Recalculate current thinking history token length before making decision
                    current_generated_thinking_tokens = len(tokenizer.encode(state['generated_thinking_history'], add_special_tokens=False))
                    thinking_limit_reached = current_generated_thinking_tokens >= think_limit_tokens - 50 

                    if state['pred_prob'] > args.threshold or thinking_limit_reached: # Third condition: already generated to </think>
                         # Probability high enough or reached thinking limit, switch to answer phase
                         state['state'] = 'needs_answer'
                         if thinking_limit_reached:
                             print(f"\nQuestion {q_idx}: Actually reached thinking limit ({current_generated_thinking_tokens}/{think_limit_tokens}). Switching to answer phase.")
                             state['too_long'] = 1
                         else:
                             print(f"\nQuestion {q_idx}: Reached early exit threshold ({state['pred_prob']:.4f} > {args.threshold}). Switching to answer phase.")
                             state['high_prob'] = 1
                    else:
                         # Probability not high enough, need more thinking
                         state['state'] = 'needs_thought_chunk'
                         if not state['current_full_sequence'].strip().endswith(continue_str) and state['thinking_steps'] != 0:
                             state['current_full_sequence'] += continue_str
                             state['generated_thinking_history'] += continue_str
                         print(f"\nQuestion {q_idx}: Early exit threshold not reached ({state['pred_prob']:.4f} <= {args.threshold}), thinking history length ({current_generated_thinking_tokens}/{think_limit_tokens}). Appending '{continue_str}' and continuing thinking.")

                elif step_type == 'answer':
                    state['generated_answer_history'] += (generated_text)
                    if last_token_id != tokenizer.eos_token_id and args.af == 1:
                        state['state'] = 'answer_forcing'
                    else:
                        state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history'] + state['generated_answer_history']
                        state['state'] = 'finished'
                        final_response_text = state['generated_thinking_history'] + state['generated_answer_history']
                        state['output_dict'] = {
                            'question': state['question_data']['question'],
                            'generated_text': final_response_text,
                            'gold_answer': state['question_data']['answer'],
                            'too_long': state['too_long'],
                            'thinking_steps': state['thinking_steps'],
                            'rep_end': state['rep_end'],
                            'high_prob': state['high_prob'],
                            'regular_end': state['regular_end'],
                            'num_trial_answer_tokens': state['num_trial_answer_tokens'],
                        }
                        if is_code:
                            state['output_dict']['question_id'] = state['question_data'].get('question_id', '')
                            state['output_dict']['extracted_code'] = extract_code_answer(final_response_text)
                        if q_idx in active_questions_indices:
                            active_questions_indices.remove(q_idx)
                            pbar.update(1)

                elif step_type == 'answer_exit':
                    state['current_full_sequence'] = state['formatted_prompt'] + state['generated_thinking_history'] + generated_text
                    state['state'] = 'finished'
                    final_response_text = state['generated_thinking_history'] + generated_text
                    state['output_dict'] = {
                        'question': state['question_data']['question'],
                        'generated_text': final_response_text,
                        'gold_answer': state['question_data']['answer'],
                        'too_long': state['too_long'],
                        'thinking_steps': state['thinking_steps'],
                        'rep_end': state['rep_end'],
                        'high_prob': state['high_prob'],
                        'regular_end': state['regular_end'],
                        'num_trial_answer_tokens': state['num_trial_answer_tokens'],
                    }
                    if is_code:
                        state['output_dict']['question_id'] = state['question_data'].get('question_id', '')
                        state['output_dict']['extracted_code'] = extract_code_answer(final_response_text)
                    if q_idx in active_questions_indices:
                        active_questions_indices.remove(q_idx)
                        pbar.update(1)

            except Exception as e:
                print(f"\nError processing batch results for question {q_idx} step '{step_type}': {e}")
                state['state'] = 'error'
                state['error_message'] = f"Error processing batch results for step '{step_type}': {e}"
                state['output_dict'] = {'question': state['question_data']['question'], 'generated_text': state['generated_thinking_history'] + state['generated_answer_history'] + "\nError: " + state['error_message'], 'gold_answer': state['question_data']['answer']}

                if q_idx in active_questions_indices:
                    active_questions_indices.remove(q_idx)
                    pbar.update(1)

    
    pbar.close()
    final_results = [state['output_dict'] for state in questions_state.values() if state['state'] in ['finished', 'error']]
    
    # Create a mapping from problem text to original index for sorting
    problem_to_index = {item['question']: i for i, item in enumerate(questions_json)}
    # Use get method to handle cases where problem text might not be found (though it shouldn't happen)
    final_results.sort(key=lambda x: problem_to_index.get(x['question'], len(questions_json)))

    print("\nAll questions processed, saving results...")
    try:
        write_jsonl(final_results, output_file)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Evaluation completed! Attempted to process {len(questions_json)} questions in total, successfully recorded {len(final_results)} results, took {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_file}")

    # Save wall time metadata
    walltime_file = os.path.join(os.path.dirname(output_file), "walltime.json")
    walltime_data = {
        "wall_time_seconds": round(elapsed_time, 2),
        "num_questions": len(final_results),
        "method": "deer",
    }
    with open(walltime_file, "w") as f:
        json.dump(walltime_data, f, indent=2)
    print(f"Wall time saved to: {walltime_file}")

if __name__ == "__main__":
    #set_seeds(42)
    main()