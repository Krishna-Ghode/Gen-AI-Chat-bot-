import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

SYSTEM_PROMPTS = {
    "Email Generator": """You are a professional email writing assistant.
Create emails based on the user's requirements.
- Include a suitable subject line when appropriate.
- Use a professional greeting, clear body, and suitable closing.
- Do not invent names, dates, achievements, or facts.
- Use placeholders when important details are missing.
Return only the requested email unless asked for explanation.""",
    "Report Generator": """You are an expert report writing assistant.
Create clear, structured and informative reports. Use relevant sections such as
Title, Introduction, Objectives, Findings, Discussion, Conclusion, and Recommendations.
Adapt the structure to the request. Do not invent statistics, research findings, or sources.
Clearly identify assumptions and missing information.""",
    "Technical Explanation": """You are an experienced technical educator and software engineer.
Explain technical topics accurately and clearly at the requested level.
When relevant, include a definition, how it works, a practical example, code if requested,
advantages, limitations, applications, and a short summary.
Do not fabricate APIs, benchmark results, or references.""",
    "General Assistant": """You are a helpful, accurate and professional AI assistant.
Understand the user's intent and answer directly. Break complex topics into steps.
Use examples when useful and acknowledge uncertainty instead of inventing facts."""
}

def load_model():
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    print("Model loaded.")
    return tokenizer, model

tokenizer, model = load_model()

def generate_response(message, history, task, tone, audience, temperature, max_new_tokens):
    if not message or not message.strip():
        return "Please enter a message or describe what you want to generate."

    system_prompt = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["General Assistant"])
    system_prompt += f"""
Additional instructions:
- Desired tone: {tone}
- Target audience or expertise level: {audience}
- Follow the user's requested format and length.
- Prioritize clarity, relevance, and factual accuracy.
- If details are missing, use placeholders or state assumptions.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Gradio ChatInterface with type="messages" supplies role/content dictionaries.
    for item in (history or [])[-6:]:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True
    ).to(model.device)

    temperature = float(temperature)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=max(temperature, 0.01),
            top_p=0.9,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response or "The model did not generate a response. Please try again."

with gr.Blocks(title="GenAI Writing Assistant") as demo:
    gr.Markdown("""
    # GenAI Writing Assistant
    Generate emails, reports, and technical explanations using an instruction-tuned LLM.

    **Model:** Qwen2.5-3B-Instruct · **Framework:** Hugging Face Transformers
    """)
    with gr.Row():
        task = gr.Dropdown(
            choices=list(SYSTEM_PROMPTS.keys()),
            value="Email Generator",
            label="Choose task"
        )
        tone = gr.Dropdown(
            choices=["Professional", "Formal", "Friendly", "Concise", "Detailed", "Academic"],
            value="Professional",
            label="Writing tone"
        )
        audience = gr.Dropdown(
            choices=["Beginner", "Intermediate", "Expert", "Professional"],
            value="Intermediate",
            label="Target audience"
        )

    temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Creativity (temperature)")
    max_new_tokens = gr.Slider(100, 1200, value=600, step=100, label="Maximum response tokens")

    gr.ChatInterface(
        fn=generate_response,
        type="messages",
        additional_inputs=[task, tone, audience, temperature, max_new_tokens],
        examples=[
            ["Write a formal email requesting an internship opportunity."],
            ["Create a report on applications of artificial intelligence."],
            ["Explain how a convolutional neural network works."],
            ["Explain REST APIs with a Python example."]
        ]
    )

if __name__ == "__main__":
    demo.launch()
