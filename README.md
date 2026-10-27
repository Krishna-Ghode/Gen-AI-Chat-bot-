# GenAI Writing Assistant

A Python Generative AI chatbot that generates professional emails, structured reports, and technical explanations using a pretrained instruction-tuned LLM and prompt engineering.

## Features
- Email Generator
- Report Generator
- Technical Explanation Generator
- General Assistant
- Adjustable tone, target audience, temperature, and response length
- Conversation context
- Gradio web interface

## Technology Stack
- Python
- Qwen2.5-3B-Instruct
- Hugging Face Transformers
- PyTorch
- Gradio

## Run locally
Python 3.10+ is recommended.

```bash
git clone <your-repository-url>
cd genai-writing-assistant
pip install -r requirements.txt
python app.py
```

The first run downloads the model from Hugging Face. A GPU is recommended; available RAM/VRAM determines whether the model can run. Once launched, open the local Gradio URL printed in the terminal.

## Run in Google Colab
1. Upload `app.py` and `requirements.txt` to a Colab notebook, or clone this repository.
2. Run `!pip install -r requirements.txt`
3. Run `!python app.py` (or paste the code into a cell and launch the app).

For a shareable temporary Gradio link in Colab, change `demo.launch()` in `app.py` to `demo.launch(share=True)`.

## How it works
1. The user chooses a task and enters a request.
2. A task-specific system prompt sets the model's role and output requirements.
3. Transformers formats the conversation using the model's chat template.
4. The LLM generates a response.
5. Gradio displays the result.

## Notes
- This project uses a pretrained model; it does not fine-tune the LLM.
- Generated content may be inaccurate. Review important emails, reports, and technical answers.
- Do not enter private or confidential information into a public demo.
