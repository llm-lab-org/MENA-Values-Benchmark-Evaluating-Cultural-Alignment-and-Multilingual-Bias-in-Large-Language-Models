import pandas as pd
import os
from tqdm import tqdm
import transformers
import torch
import numpy as np
import re
from collections import defaultdict
from torch.cuda import amp
import gc

hf_token = ''


def setup_model():
    print("Setting up the model with optimizations...")
    model_id = "meta-llama/Llama-3.1-8B-Instruct"  # this can be any model we want

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        padding_side="left",
        token=hf_token
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=True,
        token=hf_token
    )

    model.eval()

    return model, tokenizer


def extract_number(text, valid_options):
    sorted_options = sorted(valid_options, key=len, reverse=True)
    for option in sorted_options:
        pattern = r'\b' + re.escape(option) + r'\b'
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    for option in sorted_options:
        if option in text:
            return option

    return text.strip()


def batch_generate_with_logits(model, tokenizer, prompts, min_values, max_values, batch_size=32):
    all_results = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_mins = min_values[i:i + batch_size]
        batch_maxs = max_values[i:i + batch_size]

        print(f"Processing batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}")

        # Determine valid options for each prompt in the batch
        valid_options_batch = []
        formatted_prompts_batch = []

        for j, (prompt, min_val, max_val) in enumerate(zip(batch_prompts, batch_mins, batch_maxs)):
            valid_options = [str(i) for i in range(int(min_val), int(max_val) + 1)]
            valid_options_batch.append(valid_options)

            # Format the prompt for the model
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_prompts_batch.append(formatted_prompt)

        # Process each prompt individually to ensure correct response extraction
        batch_results = []
        for j, (formatted_prompt, valid_options) in enumerate(zip(formatted_prompts_batch, valid_options_batch)):
            try:
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

                with torch.no_grad(), amp.autocast(dtype=torch.bfloat16):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=15,
                        return_dict_in_generate=True,
                        output_scores=True,
                        temperature=0.0,  # Ensures purely deterministic outputs
                        do_sample=False  # Ensures no randomness is applied
                    )

                # Get the complete generated text
                generated_tokens = outputs.sequences[0]
                full_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Extract the model's response (after the prompt) - crucial for correct output
                response_start = len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True))
                generated_text = full_text[response_start:].strip()

                # Extract just the numeric response using regex with valid options
                extracted_number = extract_number(generated_text, valid_options)

                # Decode just the first few tokens individually
                first_tokens = []
                for k in range(min(3, len(outputs.sequences[0]) - len(inputs.input_ids[0]))):
                    token_idx = len(inputs.input_ids[0]) + k
                    token_id = outputs.sequences[0][token_idx].item()
                    token_text = tokenizer.decode([token_id])
                    first_tokens.append((k, token_id, token_text))

                # Print first tokens for debugging
                token_descriptions = [f"Position {k}: ID {token_id} => '{token_text}'" for k, token_id, token_text in
                                      first_tokens]

                # Find the position of our extracted answer in the generation
                answer_position = None
                for pos, token_id, token_text in first_tokens:
                    if extracted_number in token_text:
                        answer_position = pos
                        break

                # If we didn't find it, check all positions' top tokens
                if answer_position is None:
                    # Check the first few positions' top tokens
                    for k in range(min(3, len(outputs.scores))):
                        if k >= len(outputs.scores):
                            break

                        token_scores = outputs.scores[k][0]
                        top_tokens = torch.topk(token_scores, 3)
                        top_token_ids = top_tokens.indices.tolist()
                        top_token_texts = [tokenizer.decode([idx]) for idx in top_token_ids]

                        # Check if our answer is in the top tokens
                        for l, text in enumerate(top_token_texts):
                            if extracted_number in text:
                                answer_position = k
                                break

                        if answer_position is not None:
                            break

                # If still not found, use the first position for our analysis
                if answer_position is None:
                    answer_position = 0

                # Now analyze the token probabilities at the answer position
                option_probabilities = {}

                # Get the token scores at the answer position
                if answer_position < len(outputs.scores):
                    token_scores = outputs.scores[answer_position][0]
                    probabilities = torch.nn.functional.softmax(token_scores, dim=0)

                    # For each option, find the best token representation
                    for opt in valid_options:
                        token_probs = []

                        # First check if the option is directly tokenized
                        token_ids = tokenizer.encode(opt, add_special_tokens=False)

                        # If it's a single token, check its probability
                        if len(token_ids) == 1 and token_ids[0] < len(probabilities):
                            token_id = token_ids[0]
                            token_text = tokenizer.decode([token_id])
                            prob = float(probabilities[token_id].cpu().numpy())
                            logit = float(token_scores[token_id].cpu().numpy())
                            token_probs.append((prob, logit, token_id, token_text))

                        # Also check for the option with a space prefix (common tokenization)
                        space_opt = " " + opt
                        space_token_ids = tokenizer.encode(space_opt, add_special_tokens=False)
                        for token_id in space_token_ids:
                            if token_id < len(probabilities):
                                token_text = tokenizer.decode([token_id])
                                if opt in token_text:  # Make sure the token actually contains our option
                                    prob = float(probabilities[token_id].cpu().numpy())
                                    logit = float(token_scores[token_id].cpu().numpy())
                                    token_probs.append((prob, logit, token_id, token_text))

                        # Find tokens that contain this number but might not be direct encodings
                        for m, p in enumerate(probabilities):
                            if m < 1000:  # Limit search to first 1000 tokens for efficiency
                                token_text = tokenizer.decode([m])
                                if opt == token_text.strip():
                                    prob = float(p.cpu().numpy())
                                    logit = float(token_scores[m].cpu().numpy())
                                    token_probs.append((prob, logit, m, token_text))

                        # Get the highest probability variant for this option
                        if token_probs:
                            token_probs.sort(key=lambda x: x[0], reverse=True)
                            best_prob, best_logit, best_id, best_text = token_probs[0]
                            option_probabilities[opt] = {
                                "probability": best_prob,
                                "logit": best_logit,
                                "token_id": best_id,
                                "token_text": best_text
                            }
                        else:
                            option_probabilities[opt] = {
                                "probability": 0.0,
                                "logit": -1000.0,
                                "token_id": -1,
                                "token_text": "not_found"
                            }

                # Normalize the probabilities to sum to 1
                total_prob = sum(option_probabilities[opt]["probability"] for opt in valid_options)
                if total_prob > 0:
                    for opt in valid_options:
                        option_probabilities[opt]["normalized_probability"] = option_probabilities[opt][
                                                                                  "probability"] / total_prob
                else:
                    # If no probabilities found, distribute equally
                    for opt in valid_options:
                        option_probabilities[opt]["normalized_probability"] = 1.0 / len(valid_options)

                # Normalized probabilities dictionary for easy access
                norm_probs = {opt: option_probabilities[opt]["normalized_probability"] for opt in valid_options}

                # Store the final result
                result = {
                    "generated_text": generated_text,
                    "extracted_number": extracted_number,
                    "option_probabilities": option_probabilities,
                    "normalized_probabilities": norm_probs,
                    "answer_position": answer_position,
                    "first_tokens": token_descriptions
                }
                batch_results.append(result)

            except Exception as e:
                import traceback
                traceback.print_exc()

                batch_results.append({
                    "generated_text": f"Error: {str(e)}",
                    "extracted_number": "Error",
                    "option_probabilities": {},
                    "normalized_probabilities": {},
                    "answer_position": -1,
                    "first_tokens": []
                })

            if j % 4 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results.extend(batch_results)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return all_results


def process_excel_file_batch(file_path, model, tokenizer, batch_size=32, model_name=""):
    print(f"Reading Excel file: {file_path}")
    excel_file = pd.ExcelFile(file_path)
    sheet_names = excel_file.sheet_names

    updated_dfs = {}

    for sheet_name in tqdm(sheet_names, desc="Processing sheets"):
        print(f"\nProcessing sheet: {sheet_name}")
        df = pd.read_excel(file_path, sheet_name=sheet_name)

        columns = df.columns.tolist()

        for col in tqdm(columns, desc=f"Processing columns in {sheet_name}"):
            if col in columns_to_skip:

            column_data = df[col].dropna()
            valid_indices = df[col].dropna().index
            prompts = []
            min_values = []
            max_values = []

            for idx in valid_indices:
                entry = df.loc[idx, col]
                row_min = str(df.loc[idx, "Min"])
                row_max = str(df.loc[idx, "MAX"])

                prompts.append(str(entry))
                min_values.append(row_min)
                max_values.append(row_max)

            adaptive_batch_size = min(batch_size, 32)

            batch_results = batch_generate_with_logits(
                model, tokenizer, prompts, min_values, max_values, batch_size=adaptive_batch_size
            )

            full_result_series = pd.Series(index=df.index)
            number_result_series = pd.Series(index=df.index)
            logits_result_series = pd.Series(index=df.index)
            norm_probs_series = pd.Series(index=df.index)

            for i, idx in enumerate(valid_indices):
                if i < len(batch_results):
                    result = batch_results[i]
                    full_result_series[idx] = result["generated_text"]
                    number_result_series[idx] = result["extracted_number"]
                    compact_result = {
                        "extracted_answer": result["extracted_number"],
                        "answer_position": result["answer_position"],
                        "normalized_probabilities": result["normalized_probabilities"],
                        "option_details": {
                            opt: {
                                "probability": result["option_probabilities"][opt]["probability"],
                                "normalized_probability": result["option_probabilities"][opt]["normalized_probability"],
                                "token_text": result["option_probabilities"][opt]["token_text"]
                            }
                            for opt in result["option_probabilities"]
                        }
                    }
                    logits_result_series[idx] = str(compact_result)
                    norm_probs_series[idx] = str(result["normalized_probabilities"])

            df[f"{col}_{model_name}_full_answer"] = full_result_series
            df[f"{col}_{model_name}_extracted_number"] = number_result_series
            df[f"{col}_{model_name}_analysis"] = logits_result_series
            df[f"{col}_{model_name}_normalized_probs"] = norm_probs_series

            output_filename = f"{sheet_name}_{col}_answers.xlsx"
            df.to_excel(output_filename, index=False)
            print(f"  Saved intermediate result for column '{col}' as '{output_filename}'")

        updated_dfs[sheet_name] = df
        sheet_output_filename = f"{sheet_name}_updated.xlsx"
        df.to_excel(sheet_output_filename, index=False)
        print(f"Completed processing sheet '{sheet_name}' and saved as '{sheet_output_filename}'")

    with pd.ExcelWriter("updated_batch.xlsx") as writer:
        for sheet_name, df in updated_dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    torch.set_num_threads(2)  # Limit CPU threads to avoid oversubscription

    model, tokenizer = setup_model()

    file_path = ""
    process_excel_file_batch(file_path, model, tokenizer, batch_size=32)