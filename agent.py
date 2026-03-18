"""BabyVision solver — visual reasoning on early visual understanding tasks.

Takes a JSON task on stdin (question, image_path, ans_type, options), prints the answer on stdout.
Saves full LLM trajectory to eval_results/trajectories/<index>.json if EVAL_TRAJECTORY_DIR is set.
"""

import sys
import os
import json
import base64
import re
import io

from openai import OpenAI
from PIL import Image


def load_image_b64(image_path: str, min_size: int = 768) -> str:
    """Load image, upscale if too small, return base64."""
    img = Image.open(image_path)
    w, h = img.size
    # Upscale small images so the model can see more detail
    if max(w, h) < min_size:
        scale = min_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def solve(question: str, image_path: str, ans_type: str, options: list) -> str:
    client = OpenAI()

    img_b64 = load_image_b64(image_path)

    model = os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")

    # Step 1: Describe the image in detail
    describe_messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "Describe this image in detail. Focus on: the layout/grid structure, all visual elements (shapes, colors, patterns, numbers, letters), positions of elements, any differences or similarities between elements, and any spatial relationships. Be thorough and precise."},
        ]},
    ]

    desc_response = client.chat.completions.create(
        model=model,
        messages=describe_messages,
        temperature=0,
        max_completion_tokens=1024,
    )
    description = desc_response.choices[0].message.content.strip()

    # Step 2: Answer using the description + image
    if ans_type == "choice" and options:
        opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        answer_prompt = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Options:
{opts}

Think step by step, then give your final answer as ONLY the option number (1, 2, 3, or 4). Put your final answer on the last line."""
    else:
        answer_prompt = f"""Here is a detailed description of the image:
{description}

Now answer this question about the image:
{question}

Think step by step, then give your final answer in the exact format requested. Put your final answer on the last line, with ONLY the answer value and nothing else."""

    answer_messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": answer_prompt},
        ]},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=answer_messages,
        temperature=0,
        max_completion_tokens=1024,
    )

    raw_output = response.choices[0].message.content.strip()

    # Extract the last line as the final answer
    lines = [l.strip() for l in raw_output.split("\n") if l.strip()]
    answer = lines[-1] if lines else raw_output

    # Clean up formatting
    # Remove spaces after commas in coordinate-style answers
    answer = re.sub(r',\s+', ',', answer)
    # Remove spaces after hyphens in pair-style answers (like "1-F, 2-D" -> "1-F,2-D")
    # But keep hyphens within pairs (1-F stays 1-F)
    answer = re.sub(r'\s*,\s*', ',', answer)
    # Strip any trailing period
    answer = answer.rstrip('.')

    # For choice questions, extract just the number
    if ans_type == "choice":
        m = re.search(r'\b([0-4])\b', answer)
        if m:
            answer = m.group(1)

    # Save trajectory if requested
    traj_dir = os.environ.get("EVAL_TRAJECTORY_DIR")
    idx = os.environ.get("EVAL_INDEX")
    if traj_dir and idx is not None:
        os.makedirs(traj_dir, exist_ok=True)
        trajectory = {
            "index": int(idx),
            "model": model,
            "description": description,
            "question": question,
            "image_path": image_path,
            "ans_type": ans_type,
            "options": options,
            "raw_response": raw_output,
            "parsed_answer": answer,
            "usage_describe": {
                "prompt_tokens": desc_response.usage.prompt_tokens if desc_response.usage else None,
                "completion_tokens": desc_response.usage.completion_tokens if desc_response.usage else None,
            },
            "usage_answer": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
            },
        }
        with open(os.path.join(traj_dir, f"{idx}.json"), "w") as f:
            json.dump(trajectory, f, indent=2)

    return answer


if __name__ == "__main__":
    data = json.loads(sys.stdin.read().strip())
    print(solve(data["question"], data["image_path"], data["ans_type"], data.get("options", [])))
