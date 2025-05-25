Qwen3-14B
Chat
Qwen3 Highlights
Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.
Model Overview
Qwen3-14B has the following features:

Type: Causal Language Models
Training Stage: Pretraining & Post-training
Number of Parameters: 14.8B
Number of Paramaters (Non-Embedding): 13.2B
Number of Layers: 40
Number of Attention Heads (GQA): 40 for Q and 8 for KV
Context Length: 32,768 natively and 131,072 tokens with YaRN.
For more details, including benchmark evaluation, hardware requirements, and inference performance, please refer to our blog, GitHub, and Documentation.

Quickstart
The code of Qwen3 has been in the latest Hugging Face transformers and we advise you to use the latest version of transformers.

With transformers<4.51.0, you will encounter the following error:

KeyError: 'qwen3'

The following contains a code snippet illustrating how to use the model generate content based on given inputs.

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-14B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

For deployment, you can use sglang>=0.4.6.post1 or vllm>=0.8.5 or to create an OpenAI-compatible API endpoint:

SGLang:
python -m sglang.launch_server --model-path Qwen/Qwen3-14B --reasoning-parser qwen3

vLLM:
vllm serve Qwen/Qwen3-14B --enable-reasoning --reasoning-parser deepseek_r1

For local use, applications such as Ollama, LMStudio, MLX-LM, llama.cpp, and KTransformers have also supported Qwen3.

Switching Between Thinking and Non-Thinking Mode
The enable_thinking switch is also available in APIs created by SGLang and vLLM. Please refer to our documentation for SGLang and vLLM users.

enable_thinking=True
By default, Qwen3 has thinking capabilities enabled, similar to QwQ-32B. This means the model will use its reasoning abilities to enhance the quality of generated responses. For example, when explicitly setting enable_thinking=True or leaving it as the default value in tokenizer.apply_chat_template, the model will engage its thinking mode.

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # True is the default value for enable_thinking
)

In this mode, the model will generate think content wrapped in a <think>...</think> block, followed by the final response.

For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0 (the default setting in generation_config.json). DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions. For more detailed guidance, please refer to the Best Practices section.

enable_thinking=False
We provide a hard switch to strictly disable the model's thinking behavior, aligning its functionality with the previous Qwen2.5-Instruct models. This mode is particularly useful in scenarios where disabling thinking is essential for enhancing efficiency.

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False  # Setting enable_thinking=False disables thinking mode
)

In this mode, the model will not generate any think content and will not include a <think>...</think> block.

For non-thinking mode, we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0. For more detailed guidance, please refer to the Best Practices section.

Advanced Usage: Switching Between Thinking and Non-Thinking Modes via User Input
We provide a soft switch mechanism that allows users to dynamically control the model's behavior when enable_thinking=True. Specifically, you can add /think and /no_think to user prompts or system messages to switch the model's thinking mode from turn to turn. The model will follow the most recent instruction in multi-turn conversations.

Here is an example of a multi-turn conversation:

from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenChatbot:
    def __init__(self, model_name="Qwen/Qwen3-14B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.history = []

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt")
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

# Example Usage
if __name__ == "__main__":
    chatbot = QwenChatbot()

    # First input (without /think or /no_think tags, thinking mode is enabled by default)
    user_input_1 = "How many r's in strawberries?"
    print(f"User: {user_input_1}")
    response_1 = chatbot.generate_response(user_input_1)
    print(f"Bot: {response_1}")
    print("----------------------")

    # Second input with /no_think
    user_input_2 = "Then, how many r's in blueberries? /no_think"
    print(f"User: {user_input_2}")
    response_2 = chatbot.generate_response(user_input_2)
    print(f"Bot: {response_2}") 
    print("----------------------")

    # Third input with /think
    user_input_3 = "Really? /think"
    print(f"User: {user_input_3}")
    response_3 = chatbot.generate_response(user_input_3)
    print(f"Bot: {response_3}")

For API compatibility, when enable_thinking=True, regardless of whether the user uses /think or /no_think, the model will always output a block wrapped in <think>...</think>. However, the content inside this block may be empty if thinking is disabled. When enable_thinking=False, the soft switches are not valid. Regardless of any /think or /no_think tags input by the user, the model will not generate think content and will not include a <think>...</think> block.

Agentic Use
Qwen3 excels in tool calling capabilities. We recommend using Qwen-Agent to make the best use of agentic ability of Qwen3. Qwen-Agent encapsulates tool-calling templates and tool-calling parsers internally, greatly reducing coding complexity.

To define the available tools, you can use the MCP configuration file, use the integrated tool of Qwen-Agent, or integrate other tools by yourself.

from qwen_agent.agents import Assistant

# Define LLM
llm_cfg = {
    'model': 'Qwen3-14B',

    # Use the endpoint provided by Alibaba Model Studio:
    # 'model_type': 'qwen_dashscope',
    # 'api_key': os.getenv('DASHSCOPE_API_KEY'),

    # Use a custom endpoint compatible with OpenAI API:
    'model_server': 'http://localhost:8000/v1',  # api_base
    'api_key': 'EMPTY',

    # Other parameters:
    # 'generate_cfg': {
    #         # Add: When the response content is `<think>this is the thought</think>this is the answer;
    #         # Do not add: When the response has been separated by reasoning_content and content.
    #         'thought_in_content': True,
    #     },
}

# Define Tools
tools = [
    {'mcpServers': {  # You can specify the MCP configuration file
            'time': {
                'command': 'uvx',
                'args': ['mcp-server-time', '--local-timezone=Asia/Shanghai']
            },
            "fetch": {
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
  'code_interpreter',  # Built-in tools
]

# Define Agent
bot = Assistant(llm=llm_cfg, function_list=tools)

# Streaming generation
messages = [{'role': 'user', 'content': 'https://qwenlm.github.io/blog/ Introduce the latest developments of Qwen'}]
for responses in bot.run(messages=messages):
    pass
print(responses)

Processing Long Texts
Qwen3 natively supports context lengths of up to 32,768 tokens. For conversations where the total length (including both input and output) significantly exceeds this limit, we recommend using RoPE scaling techniques to handle long texts effectively. We have validated the model's performance on context lengths of up to 131,072 tokens using the YaRN method.

YaRN is currently supported by several inference frameworks, e.g., transformers and llama.cpp for local use, vllm and sglang for deployment. In general, there are two approaches to enabling YaRN for supported frameworks:

Modifying the model files: In the config.json file, add the rope_scaling fields:

{
    ...,
    "rope_scaling": {
        "rope_type": "yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 32768
    }
}

For llama.cpp, you need to regenerate the GGUF file after the modification.

Passing command line arguments:

For vllm, you can use

vllm serve ... --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072  

For sglang, you can use

python -m sglang.launch_server ... --json-model-override-args '{"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}}'

For llama-server from llama.cpp, you can use

llama-server ... --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768

If you encounter the following warning

Unrecognized keys in `rope_scaling` for 'rope_type'='yarn': {'original_max_position_embeddings'}

please upgrade transformers>=4.51.0.

All the notable open-source frameworks implement static YaRN, which means the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts. We advise adding the rope_scaling configuration only when processing long contexts is required. It is also recommended to modify the factor as needed. For example, if the typical context length for your application is 65,536 tokens, it would be better to set factor as 2.0.

The default max_position_embeddings in config.json is set to 40,960. This allocation includes reserving 32,768 tokens for outputs and 8,192 tokens for typical prompts, which is sufficient for most scenarios involving short text processing. If the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN in this scenario, as it may potentially degrade model performance.

The endpoint provided by Alibaba Model Studio supports dynamic YaRN by default and no extra configuration is needed.

Best Practices
To achieve optimal performance, we recommend the following settings:

Sampling Parameters:

For thinking mode (enable_thinking=True), use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0. DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
For supported frameworks, you can adjust the presence_penalty parameter between 0 and 2 to reduce endless repetitions. However, using a higher value may occasionally result in language mixing and a slight decrease in model performance.
Adequate Output Length: We recommend using an output length of 32,768 tokens for most queries. For benchmarking on highly complex problems, such as those found in math and programming competitions, we suggest setting the max output length to 38,912 tokens. This provides the model with sufficient space to generate detailed and comprehensive responses, thereby enhancing its overall performance.

Standardize Output Format: We recommend using prompts to standardize model outputs when benchmarking.

Math Problems: Include "Please reason step by step, and put your final answer within \boxed{}." in the prompt.
Multiple-Choice Questions: Add the following JSON structure to the prompt to standardize responses: "Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."
No Thinking Content in History: In multi-turn conversations, the historical model output should only include the final output part and does not need to include the thinking content. It is implemented in the provided chat template in Jinja2. However, for frameworks that do not directly use the Jinja2 chat template, it is up to the developers to ensure that the best practice is followed.

Citation
If you find our work helpful, feel free to give us a cite.

@misc{qwen3technicalreport,
      title={Qwen3 Technical Report}, 
      author={Qwen Team},
      year={2025},
      eprint={2505.09388},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.09388}, 
}






----------------------------------------
Bee1reason-arabic-Qwen-14B: A Qwen3 14B Model Fine-tuned for Arabic Logical Reasoning
Model Overview
Bee1reason-arabic-Qwen-14B is a Large Language Model (LLM) fine-tuned from the unsloth/Qwen3-14B base model (which itself is based on Qwen/Qwen2-14B). This model has been specifically tailored to enhance logical and deductive reasoning capabilities in the Arabic language, while also maintaining its general conversational abilities. The fine-tuning process utilized LoRA (Low-Rank Adaptation) with the Unsloth library for high training efficiency. The LoRA weights were then merged with the base model to produce this standalone 16-bit (float16) precision model.

Key Features:

Built on unsloth/Qwen3-14B: Leverages the power and performance of the Qwen3 14-billion parameter base model.
Fine-tuned for Arabic Logical Reasoning: Trained on a dataset containing Arabic logical reasoning tasks.
Conversational Format: The model follows a conversational format, expecting user and assistant roles. It was trained on data that may include "thinking steps" (often within <think>...</think> tags) before providing the final answer, which is beneficial for tasks requiring explanation or complex inference.
Unsloth Efficiency: The Unsloth library was used for the fine-tuning process, enabling faster training and reduced GPU memory consumption.
Merged 16-bit Model: The final weights are a full float16 precision model, ready for direct use without needing to apply LoRA adapters to a separate base model.
Training Data
The model was primarily fine-tuned on a custom Arabic logical reasoning dataset, beetlware/arabic-reasoning-dataset-logic, available on the Hugging Face Hub. This dataset includes tasks variés types of reasoning (deduction, induction, abduction), with each task comprising the question text, a proposed answer, and a detailed solution including thinking steps.

This data was converted into a conversational format for training, typically with:

User Role: Containing the problem/question text.
Assistant Role: Containing the detailed solution, including thinking steps (often within <think>...</think> tags) followed by the final answer.
Fine-tuning Details
Base Model: unsloth/Qwen3-14B
Fine-tuning Technique: LoRA (Low-Rank Adaptation)
r (rank): 32
lora_alpha: 32
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout: 0
bias: "none"
Libraries Used: Unsloth (for efficient model loading and PEFT application) and Hugging Face TRL (SFTTrainer)
Max Sequence Length (max_seq_length): 2048 tokens
Training Parameters (example from notebook):
per_device_train_batch_size: 2
gradient_accumulation_steps: 4 (simulating a total batch size of 8)
warmup_steps: 5
max_steps: 30 (in the notebook, adjustable for a full run)
learning_rate: 2e-4 (recommended to reduce to 2e-5 for longer training runs)
optim: "adamw_8bit"
Final Save: LoRA weights were merged with the base model and saved in merged_16bit (float16) precision.
How to Use (with Transformers)
Since this is a merged 16-bit model, you can load and use it directly with the transformers library:

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

model_id = "beetlware/Bee1reason-arabic-Qwen-14B"

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # or torch.float16 if bfloat16 is not supported
    device_map="auto", # Distributes the model on available devices (GPU/CPU)
)

# Ensure the model is in evaluation mode for inference
model.eval()

--- Example for Inference with Thinking Steps (if the model was trained to produce them) ---
Qwen3 models expect special tags for thinking ...
To enable thinking mode during inference (if supported by the fine-tuned model):
You might need to craft the prompt to ask the model to think.
Unsloth-trained Qwen3 models often respond to enable_thinking in tokenizer.apply_chat_template.
For a merged model, its ability to show depends on the training data.
user_prompt_with_thinking_request = "استخدم التفكير المنطقي خطوة بخطوة: إذا كان لدي 4 تفاحات والشجرة فيها 20 تفاحة، فكم تفاحة لدي إجمالاً؟" # "Use step-by-step logical thinking: If I have 4 apples and the tree has 20 apples, how many apples do I have in total?"

messages_with_thinking = [
    {"role": "user", "content": user_prompt_with_thinking_request}
]

# Apply the chat template
# Qwen3 uses a specific chat template. tokenizer.apply_chat_template is the correct way to format it.
chat_prompt_with_thinking = tokenizer.apply_chat_template(
    messages_with_thinking,
    tokenize=False,
    add_generation_prompt=True # Important for adding the assistant's generation prompt
)

inputs_with_thinking = tokenizer(chat_prompt_with_thinking, return_tensors="pt").to(model.device)

print("\n--- Inference with Thinking Request (Example) ---")
streamer_think = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad(): # Important to disable gradients during inference
    outputs_think = model.generate(
        **inputs_with_thinking,
        max_new_tokens=512,
        temperature=0.6, # Recommended settings for reasoning by Qwen team
        top_p=0.95,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_think
    )

# --- Example for Normal Inference (Conversation without explicit thinking request) ---
user_prompt_normal = "ما هي عاصمة مصر؟" # "What is the capital of Egypt?"
messages_normal = [
    {"role": "user", "content": user_prompt_normal}
]

chat_prompt_normal = tokenizer.apply_chat_template(
    messages_normal,
    tokenize=False,
    add_generation_prompt=True
)
inputs_normal = tokenizer(chat_prompt_normal, return_tensors="pt").to(model.device)

print("\n\n--- Normal Inference (Example) ---")
streamer_normal = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad():
    outputs_normal = model.generate(
        **inputs_normal,
        max_new_tokens=100,
        temperature=0.7, # Recommended settings for normal chat
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_normal
    )

Usage with VLLM (for High-Throughput Scaled Inference)
VLLM is a library for fast LLM inference. Since you saved the model as merged_16bit, it can be used with VLLM.

Install VLLM:

pip install vllm

(VLLM installation might have specific CUDA and PyTorch version requirements. Refer to the VLLM documentation for the latest installation prerequisites.)

Run the VLLM OpenAI-Compatible Server: You can serve the model using VLLM's OpenAI-compatible API server, making it easy to integrate into existing applications.
python -m vllm.entrypoints.openai.api_server \
    --model beetlware/Bee1reason-arabic-Qwen-14B \
    --tokenizer beetlware/Bee1reason-arabic-Qwen-14B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    # --tensor-parallel-size N  # If you have multiple GPUs
    # --gpu-memory-utilization 0.9 # To adjust GPU memory usage

Replace --dtype bfloat16 with float16 if needed.
max-model-len should match the max_seq_length you used.
Send Requests to the VLLM Server: Once the server is running (typically on http://localhost:8000), you can send requests using any OpenAI-compatible client, like the openai library:

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1", # VLLM server address
    api_key="dummy_key" # VLLM doesn't require an actual API key by default
)

completion = client.chat.completions.create(
    model="beetlware/Bee1reason-arabic-Qwen-14B", # Model name as specified in VLLM
    messages=[
        {"role": "user", "content": "اشرح نظرية النسبية العامة بكلمات بسيطة."} # "Explain the theory of general relativity in simple terms."
    ],
    max_tokens=256,
    temperature=0.7,
    stream=True # To enable streaming
)

print("Streaming response from VLLM:")
full_response = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        token = chunk.choices[0].delta.content
        print(token, end="", flush=True)
        full_response += token
print("\n--- End of stream ---")

Limitations and Potential Biases
The model's performance is highly dependent on the quality and diversity of the training data. It may exhibit biases present in the data it was trained on. Despite fine-tuning for logical reasoning, the model might still make errors on very complex or unfamiliar reasoning tasks. The model may "hallucinate" or produce incorrect information, especially for topics not well-covered in its training data. Capabilities in languages other than Arabic (if primarily trained on Arabic) might be limited.

Additional Information
Developed by: [loai abdalslam/Organization - beetleware] Upload/Release Date: [21-5-2025] Contact / Issue Reporting: [loai.abdalsalm@beetleware.com]

Beetleware :
We are a software house and digital transformation service provider that was founded six years ago and is based in Saudi Arabia.

All rights reserved@2025

Our Offices

KSA Office (+966) 54 597 3282 ahmed.taha@beetleware.com

Egypt Office (+2) 010 67 256 306 ahmed.abullah@beetleware.com

Oman Office (+968) 9522 8632

Uploaded model
Developed by: beetlware AI Team
License: apache-2.0
Finetuned from model : unsloth/qwen3-14b-unsloth-bnb-4bit
This qwen3 model was trained 2x faster with Unsloth and Huggingface's TRL library.


-----------------------------------------------------------
Bee1reason-arabic-Qwen-14B: A Qwen3 14B Model Fine-tuned for Arabic Logical Reasoning
Model Overview
Bee1reason-arabic-Qwen-14B is a Large Language Model (LLM) fine-tuned from the unsloth/Qwen3-14B base model (which itself is based on Qwen/Qwen2-14B). This model has been specifically tailored to enhance logical and deductive reasoning capabilities in the Arabic language, while also maintaining its general conversational abilities. The fine-tuning process utilized LoRA (Low-Rank Adaptation) with the Unsloth library for high training efficiency. The LoRA weights were then merged with the base model to produce this standalone 16-bit (float16) precision model.

Key Features:

Built on unsloth/Qwen3-14B: Leverages the power and performance of the Qwen3 14-billion parameter base model.
Fine-tuned for Arabic Logical Reasoning: Trained on a dataset containing Arabic logical reasoning tasks.
Conversational Format: The model follows a conversational format, expecting user and assistant roles. It was trained on data that may include "thinking steps" (often within <think>...</think> tags) before providing the final answer, which is beneficial for tasks requiring explanation or complex inference.
Unsloth Efficiency: The Unsloth library was used for the fine-tuning process, enabling faster training and reduced GPU memory consumption.
Merged 16-bit Model: The final weights are a full float16 precision model, ready for direct use without needing to apply LoRA adapters to a separate base model.
Training Data
The model was primarily fine-tuned on a custom Arabic logical reasoning dataset, beetlware/arabic-reasoning-dataset-logic, available on the Hugging Face Hub. This dataset includes tasks variés types of reasoning (deduction, induction, abduction), with each task comprising the question text, a proposed answer, and a detailed solution including thinking steps.

This data was converted into a conversational format for training, typically with:

User Role: Containing the problem/question text.
Assistant Role: Containing the detailed solution, including thinking steps (often within <think>...</think> tags) followed by the final answer.
Fine-tuning Details
Base Model: unsloth/Qwen3-14B
Fine-tuning Technique: LoRA (Low-Rank Adaptation)
r (rank): 32
lora_alpha: 32
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout: 0
bias: "none"
Libraries Used: Unsloth (for efficient model loading and PEFT application) and Hugging Face TRL (SFTTrainer)
Max Sequence Length (max_seq_length): 2048 tokens
Training Parameters (example from notebook):
per_device_train_batch_size: 2
gradient_accumulation_steps: 4 (simulating a total batch size of 8)
warmup_steps: 5
max_steps: 30 (in the notebook, adjustable for a full run)
learning_rate: 2e-4 (recommended to reduce to 2e-5 for longer training runs)
optim: "adamw_8bit"
Final Save: LoRA weights were merged with the base model and saved in merged_16bit (float16) precision.
How to Use (with Transformers)
Since this is a merged 16-bit model, you can load and use it directly with the transformers library:

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

model_id = "beetlware/Bee1reason-arabic-Qwen-14B"

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # or torch.float16 if bfloat16 is not supported
    device_map="auto", # Distributes the model on available devices (GPU/CPU)
)

# Ensure the model is in evaluation mode for inference
model.eval()

user_prompt_with_thinking_request = "استخدم التفكير المنطقي خطوة بخطوة: إذا كان لدي 4 تفاحات والشجرة فيها 20 تفاحة، فكم تفاحة لدي إجمالاً؟" # "Use step-by-step logical thinking: If I have 4 apples and the tree has 20 apples, how many apples do I have in total?"

messages_with_thinking = [
    {"role": "user", "content": user_prompt_with_thinking_request}
]

# Apply the chat template
# Qwen3 uses a specific chat template. tokenizer.apply_chat_template is the correct way to format it.
chat_prompt_with_thinking = tokenizer.apply_chat_template(
    messages_with_thinking,
    tokenize=False,
    add_generation_prompt=True # Important for adding the assistant's generation prompt
)

inputs_with_thinking = tokenizer(chat_prompt_with_thinking, return_tensors="pt").to(model.device)

print("\n--- Inference with Thinking Request (Example) ---")
streamer_think = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad(): # Important to disable gradients during inference
    outputs_think = model.generate(
        **inputs_with_thinking,
        max_new_tokens=512,
        temperature=0.6, # Recommended settings for reasoning by Qwen team
        top_p=0.95,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_think
    )

# --- Example for Normal Inference (Conversation without explicit thinking request) ---
user_prompt_normal = "ما هي عاصمة مصر؟" # "What is the capital of Egypt?"
messages_normal = [
    {"role": "user", "content": user_prompt_normal}
]

chat_prompt_normal = tokenizer.apply_chat_template(
    messages_normal,
    tokenize=False,
    add_generation_prompt=True
)
inputs_normal = tokenizer(chat_prompt_normal, return_tensors="pt").to(model.device)

print("\n\n--- Normal Inference (Example) ---")
streamer_normal = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad():
    outputs_normal = model.generate(
        **inputs_normal,
        max_new_tokens=100,
        temperature=0.7, # Recommended settings for normal chat
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_normal
    )

Usage with VLLM (for High-Throughput Scaled Inference)
VLLM is a library for fast LLM inference. Since you saved the model as merged_16bit, it can be used with VLLM.

Install VLLM:

pip install vllm

(VLLM installation might have specific CUDA and PyTorch version requirements. Refer to the VLLM documentation for the latest installation prerequisites.)

Run the VLLM OpenAI-Compatible Server: You can serve the model using VLLM's OpenAI-compatible API server, making it easy to integrate into existing applications.
python -m vllm.entrypoints.openai.api_server \
    --model beetlware/Bee1reason-arabic-Qwen-14B \
    --tokenizer beetlware/Bee1reason-arabic-Qwen-14B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    # --tensor-parallel-size N  # If you have multiple GPUs
    # --gpu-memory-utilization 0.9 # To adjust GPU memory usage

Replace --dtype bfloat16 with float16 if needed.
max-model-len should match the max_seq_length you used.
Send Requests to the VLLM Server: Once the server is running (typically on http://localhost:8000), you can send requests using any OpenAI-compatible client, like the openai library:

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1", # VLLM server address
    api_key="dummy_key" # VLLM doesn't require an actual API key by default
)

completion = client.chat.completions.create(
    model="beetlware/Bee1reason-arabic-Qwen-14B", # Model name as specified in VLLM
    messages=[
        {"role": "user", "content": "اشرح نظرية النسبية العامة بكلمات بسيطة."} # "Explain the theory of general relativity in simple terms."
    ],
    max_tokens=256,
    temperature=0.7,
    stream=True # To enable streaming
)

print("Streaming response from VLLM:")
full_response = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        token = chunk.choices[0].delta.content
        print(token, end="", flush=True)
        full_response += token
print("\n--- End of stream ---")

Limitations and Potential Biases
The model's performance is highly dependent on the quality and diversity of the training data. It may exhibit biases present in the data it was trained on. Despite fine-tuning for logical reasoning, the model might still make errors on very complex or unfamiliar reasoning tasks. The model may "hallucinate" or produce incorrect information, especially for topics not well-covered in its training data. Capabilities in languages other than Arabic (if primarily trained on Arabic) might be limited.

Additional Information
Developed by: [loai abdalslam/Organization - beetleware] Upload/Release Date: [21-5-2025] Contact / Issue Reporting: [loai.abdalsalm@beetleware.com]

Beetleware :
We are a software house and digital transformation service provider that was founded six years ago and is based in Saudi Arabia.

All rights reserved@2025

Our Offices

KSA Office (+966) 54 597 3282 ahmed.taha@beetleware.com

Egypt Office (+2) 010 67 256 306 ahmed.abullah@beetleware.com

Oman Office (+968) 9522 8632

Uploaded model
Developed by: beetlware AI Team
License: apache-2.0
Finetuned from model : unsloth/qwen3-14b-unsloth-bnb-4bit
This qwen3 model was trained 2x faster with Unsloth and Huggingface's TRL library.


------------------------------------------------


Bee1reason-arabic-Qwen-14B: A Qwen3 14B Model Fine-tuned for Arabic Logical Reasoning
Model Overview
Bee1reason-arabic-Qwen-14B is a Large Language Model (LLM) fine-tuned from the unsloth/Qwen3-14B base model (which itself is based on Qwen/Qwen2-14B). This model has been specifically tailored to enhance logical and deductive reasoning capabilities in the Arabic language, while also maintaining its general conversational abilities. The fine-tuning process utilized LoRA (Low-Rank Adaptation) with the Unsloth library for high training efficiency. The LoRA weights were then merged with the base model to produce this standalone 16-bit (float16) precision model.

Key Features:

Built on unsloth/Qwen3-14B: Leverages the power and performance of the Qwen3 14-billion parameter base model.
Fine-tuned for Arabic Logical Reasoning: Trained on a dataset containing Arabic logical reasoning tasks.
Conversational Format: The model follows a conversational format, expecting user and assistant roles. It was trained on data that may include "thinking steps" (often within <think>...</think> tags) before providing the final answer, which is beneficial for tasks requiring explanation or complex inference.
Unsloth Efficiency: The Unsloth library was used for the fine-tuning process, enabling faster training and reduced GPU memory consumption.
Merged 16-bit Model: The final weights are a full float16 precision model, ready for direct use without needing to apply LoRA adapters to a separate base model.
Training Data
The model was primarily fine-tuned on a custom Arabic logical reasoning dataset, beetlware/arabic-reasoning-dataset-logic, available on the Hugging Face Hub. This dataset includes tasks variés types of reasoning (deduction, induction, abduction), with each task comprising the question text, a proposed answer, and a detailed solution including thinking steps.

This data was converted into a conversational format for training, typically with:

User Role: Containing the problem/question text.
Assistant Role: Containing the detailed solution, including thinking steps (often within <think>...</think> tags) followed by the final answer.
Fine-tuning Details
Base Model: unsloth/Qwen3-14B
Fine-tuning Technique: LoRA (Low-Rank Adaptation)
r (rank): 32
lora_alpha: 32
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_dropout: 0
bias: "none"
Libraries Used: Unsloth (for efficient model loading and PEFT application) and Hugging Face TRL (SFTTrainer)
Max Sequence Length (max_seq_length): 2048 tokens
Training Parameters (example from notebook):
per_device_train_batch_size: 2
gradient_accumulation_steps: 4 (simulating a total batch size of 8)
warmup_steps: 5
max_steps: 30 (in the notebook, adjustable for a full run)
learning_rate: 2e-4 (recommended to reduce to 2e-5 for longer training runs)
optim: "adamw_8bit"
Final Save: LoRA weights were merged with the base model and saved in merged_16bit (float16) precision.
How to Use (with Transformers)
Since this is a merged 16-bit model, you can load and use it directly with the transformers library:

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

model_id = "beetlware/Bee1reason-arabic-Qwen-14B"

# Load the Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the Model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # or torch.float16 if bfloat16 is not supported
    device_map="auto", # Distributes the model on available devices (GPU/CPU)
)

# Ensure the model is in evaluation mode for inference
model.eval()

--- Example for Inference with Thinking Steps (if the model was trained to produce them) ---
Qwen3 models expect special tags for thinking ...
To enable thinking mode during inference (if supported by the fine-tuned model):
You might need to craft the prompt to ask the model to think.
Unsloth-trained Qwen3 models often respond to enable_thinking in tokenizer.apply_chat_template.
For a merged model, its ability to show depends on the training data.
user_prompt_with_thinking_request = "استخدم التفكير المنطقي خطوة بخطوة: إذا كان لدي 4 تفاحات والشجرة فيها 20 تفاحة، فكم تفاحة لدي إجمالاً؟" # "Use step-by-step logical thinking: If I have 4 apples and the tree has 20 apples, how many apples do I have in total?"

messages_with_thinking = [
    {"role": "user", "content": user_prompt_with_thinking_request}
]

# Apply the chat template
# Qwen3 uses a specific chat template. tokenizer.apply_chat_template is the correct way to format it.
chat_prompt_with_thinking = tokenizer.apply_chat_template(
    messages_with_thinking,
    tokenize=False,
    add_generation_prompt=True # Important for adding the assistant's generation prompt
)

inputs_with_thinking = tokenizer(chat_prompt_with_thinking, return_tensors="pt").to(model.device)

print("\n--- Inference with Thinking Request (Example) ---")
streamer_think = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad(): # Important to disable gradients during inference
    outputs_think = model.generate(
        **inputs_with_thinking,
        max_new_tokens=512,
        temperature=0.6, # Recommended settings for reasoning by Qwen team
        top_p=0.95,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_think
    )

# --- Example for Normal Inference (Conversation without explicit thinking request) ---
user_prompt_normal = "ما هي عاصمة مصر؟" # "What is the capital of Egypt?"
messages_normal = [
    {"role": "user", "content": user_prompt_normal}
]

chat_prompt_normal = tokenizer.apply_chat_template(
    messages_normal,
    tokenize=False,
    add_generation_prompt=True
)
inputs_normal = tokenizer(chat_prompt_normal, return_tensors="pt").to(model.device)

print("\n\n--- Normal Inference (Example) ---")
streamer_normal = TextStreamer(tokenizer, skip_prompt=True)
with torch.no_grad():
    outputs_normal = model.generate(
        **inputs_normal,
        max_new_tokens=100,
        temperature=0.7, # Recommended settings for normal chat
        top_p=0.8,
        top_k=20,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer_normal
    )

Usage with VLLM (for High-Throughput Scaled Inference)
VLLM is a library for fast LLM inference. Since you saved the model as merged_16bit, it can be used with VLLM.

Install VLLM:

pip install vllm

(VLLM installation might have specific CUDA and PyTorch version requirements. Refer to the VLLM documentation for the latest installation prerequisites.)

Run the VLLM OpenAI-Compatible Server: You can serve the model using VLLM's OpenAI-compatible API server, making it easy to integrate into existing applications.
python -m vllm.entrypoints.openai.api_server \
    --model beetlware/Bee1reason-arabic-Qwen-14B \
    --tokenizer beetlware/Bee1reason-arabic-Qwen-14B \
    --dtype bfloat16 \
    --max-model-len 2048 \
    # --tensor-parallel-size N  # If you have multiple GPUs
    # --gpu-memory-utilization 0.9 # To adjust GPU memory usage

Replace --dtype bfloat16 with float16 if needed.
max-model-len should match the max_seq_length you used.
Send Requests to the VLLM Server: Once the server is running (typically on http://localhost:8000), you can send requests using any OpenAI-compatible client, like the openai library:

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1", # VLLM server address
    api_key="dummy_key" # VLLM doesn't require an actual API key by default
)

completion = client.chat.completions.create(
    model="beetlware/Bee1reason-arabic-Qwen-14B", # Model name as specified in VLLM
    messages=[
        {"role": "user", "content": "اشرح نظرية النسبية العامة بكلمات بسيطة."} # "Explain the theory of general relativity in simple terms."
    ],
    max_tokens=256,
    temperature=0.7,
    stream=True # To enable streaming
)

print("Streaming response from VLLM:")
full_response = ""
for chunk in completion:
    if chunk.choices[0].delta.content is not None:
        token = chunk.choices[0].delta.content
        print(token, end="", flush=True)
        full_response += token
print("\n--- End of stream ---")

Limitations and Potential Biases
The model's performance is highly dependent on the quality and diversity of the training data. It may exhibit biases present in the data it was trained on. Despite fine-tuning for logical reasoning, the model might still make errors on very complex or unfamiliar reasoning tasks. The model may "hallucinate" or produce incorrect information, especially for topics not well-covered in its training data. Capabilities in languages other than Arabic (if primarily trained on Arabic) might be limited.

Additional Information
Developed by: [loai abdalslam/Organization - beetleware] Upload/Release Date: [21-5-2025] Contact / Issue Reporting: [loai.abdalsalm@beetleware.com]

Beetleware :
We are a software house and digital transformation service provider that was founded six years ago and is based in Saudi Arabia.

All rights reserved@2025

Our Offices

KSA Office (+966) 54 597 3282 ahmed.taha@beetleware.com

Egypt Office (+2) 010 67 256 306 ahmed.abullah@beetleware.com

Oman Office (+968) 9522 8632

Uploaded model
Developed by: beetlware AI Team
License: apache-2.0
Finetuned from model : unsloth/qwen3-14b-unsloth-bnb-4bit
This qwen3 model was trained 2x faster with Unsloth and Huggingface's TRL library.


----------------------------------------------------------------------------


Model Information
The Llama 4 collection of models are natively multimodal AI models that enable text and multimodal experiences. These models leverage a mixture-of-experts architecture to offer industry-leading performance in text and image understanding.

These Llama 4 models mark the beginning of a new era for the Llama ecosystem. We are launching two efficient models in the Llama 4 series, Llama 4 Scout, a 17 billion parameter model with 16 experts, and Llama 4 Maverick, a 17 billion parameter model with 128 experts.

Model developer: Meta

Model Architecture: The Llama 4 models are auto-regressive language models that use a mixture-of-experts (MoE) architecture and incorporate early fusion for native multimodality.

Model Name	Training Data	Params	Input modalities	Output modalities	Context length	Token count	Knowledge cutoff
Llama 4 Scout (17Bx16E)	A mix of publicly available, licensed data and information from Meta's products and services. This includes publicly shared posts from Instagram and Facebook and people's interactions with Meta AI. Learn more in our Privacy Center.	17B (Activated) 109B (Total)	Multilingual text and image	Multilingual text and code	10M	~40T	August 2024
Llama 4 Maverick (17Bx128E)	17B (Activated) 400B (Total)	Multilingual text and image	Multilingual text and code	1M	~22T	August 2024
Supported languages: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese.

Model Release Date: April 5, 2025

Status: This is a static model trained on an offline dataset. Future versions of the tuned models may be released as we improve model behavior with community feedback.

License: A custom commercial license, the Llama 4 Community License Agreement, is available at: https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE

Where to send questions or comments about the model: Instructions on how to provide feedback or comments on the model can be found in the Llama README. For more technical information about generation parameters and recipes for how to use Llama 4 in applications, please go here.

Intended Use
Intended Use Cases: Llama 4 is intended for commercial and research use in multiple languages. Instruction tuned models are intended for assistant-like chat and visual reasoning tasks, whereas pretrained models can be adapted for natural language generation. For vision, Llama 4 models are also optimized for visual recognition, image reasoning, captioning, and answering general questions about an image. The Llama 4 model collection also supports the ability to leverage the outputs of its models to improve other models including synthetic data generation and distillation. The Llama 4 Community License allows for these use cases.

Out-of-scope: Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 4 Community License. Use in languages or capabilities beyond those explicitly referenced as supported in this model card**.

**Note:

1. Llama 4 has been trained on a broader collection of languages than the 12 supported languages (pre-training includes 200 total languages). Developers may fine-tune Llama 4 models for languages beyond the 12 supported languages provided they comply with the Llama 4 Community License and the Acceptable Use Policy. Developers are responsible for ensuring that their use of Llama 4 in additional languages is done in a safe and responsible manner.

2. Llama 4 has been tested for image understanding up to 5 input images. If leveraging additional image understanding capabilities beyond this, Developers are responsible for ensuring that their deployments are mitigated for risks and should perform additional testing and tuning tailored to their specific applications.

How to use with transformers
Please, make sure you have transformers v4.51.0 installed, or upgrade using pip install -U transformers.

from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])

Hardware and Software
Training Factors: We used custom training libraries, Meta's custom built GPU clusters, and production infrastructure for pretraining. Fine-tuning, quantization, annotation, and evaluation were also performed on production infrastructure.

Training Energy Use: Model pre-training utilized a cumulative of 7.38M GPU hours of computation on H100-80GB (TDP of 700W) type hardware, per the table below. Training time is the total GPU time required for training each model and power consumption is the peak power capacity per GPU device used, adjusted for power usage efficiency.

Training Greenhouse Gas Emissions: Estimated total location-based greenhouse gas emissions were 1,999 tons CO2eq for training. Since 2020, Meta has maintained net zero greenhouse gas emissions in its global operations and matched 100% of its electricity use with clean and renewable energy; therefore, the total market-based greenhouse gas emissions for training were 0 tons CO2eq.
Model Name	Training Time (GPU hours)	Training Power Consumption (W)	Training Location-Based Greenhouse Gas Emissions (tons CO2eq)	Training Market-Based Greenhouse Gas Emissions (tons CO2eq)
Llama 4 Scout	5.0M	700	1,354	0
Llama 4 Maverick	2.38M	700	645	0
Total	7.38M	-	1,999	0
The methodology used to determine training energy use and greenhouse gas emissions can be found here. Since Meta is openly releasing these models, the training energy use and greenhouse gas emissions will not be incurred by others.
Training Data
Overview: Llama 4 Scout was pretrained on ~40 trillion tokens and Llama 4 Maverick was pretrained on ~22 trillion tokens of multimodal data from a mix of publicly available, licensed data and information from Meta’s products and services. This includes publicly shared posts from Instagram and Facebook and people’s interactions with Meta AI.

Data Freshness: The pretraining data has a cutoff of August 2024.

Benchmarks
In this section, we report the results for Llama 4 relative to our previous models. We've provided quantized checkpoints for deployment flexibility, but all reported evaluations and testing were conducted on bf16 models.

Pre-trained models
Pre-trained models							
Category	Benchmark	# Shots	Metric	Llama 3.1 70B	Llama 3.1 405B	Llama 4 Scout	Llama 4 Maverick
Reasoning & Knowledge	MMLU	5	macro_avg/acc_char	79.3	85.2	79.6	85.5
MMLU-Pro	5	macro_avg/em	53.8	61.6	58.2	62.9
MATH	4	em_maj1@1	41.6	53.5	50.3	61.2
Code	MBPP	3	pass@1	66.4	74.4	67.8	77.6
Multilingual	TydiQA	1	average/f1	29.9	34.3	31.5	31.7
Image	ChartQA	0	relaxed_accuracy	No multimodal support		83.4	85.3
DocVQA	0	anls			89.4	91.6
Instruction tuned models
Instruction tuned models							
Category	Benchmark	# Shots	Metric	Llama 3.3 70B	Llama 3.1 405B	Llama 4 Scout	Llama 4 Maverick
Image Reasoning	MMMU	0	accuracy	No multimodal support		69.4	73.4
MMMU Pro^	0	accuracy			52.2	59.6
MathVista	0	accuracy			70.7	73.7
Image Understanding	ChartQA	0	relaxed_accuracy			88.8	90.0
DocVQA (test)	0	anls			94.4	94.4
Coding	LiveCodeBench (10/01/2024-02/01/2025)	0	pass@1	33.3	27.7	32.8	43.4
Reasoning & Knowledge	MMLU Pro	0	macro_avg/acc	68.9	73.4	74.3	80.5
GPQA Diamond	0	accuracy	50.5	49.0	57.2	69.8
Multilingual	MGSM	0	average/em	91.1	91.6	90.6	92.3
Long context	MTOB (half book) eng->kgv/kgv->eng	-	chrF	Context window is 128K		42.2/36.6	54.0/46.4
MTOB (full book) eng->kgv/kgv->eng	-	chrF			39.7/36.3	50.8/46.7
^reported numbers for MMMU Pro is the average of Standard and Vision tasks

Quantization
The Llama 4 Scout model is released as BF16 weights, but can fit within a single H100 GPU with on-the-fly int4 quantization; the Llama 4 Maverick model is released as both BF16 and FP8 quantized weights. The FP8 quantized weights fit on a single H100 DGX host while still maintaining quality. We provide code for on-the-fly int4 quantization which minimizes performance degradation as well.

Safeguards
As part of our release approach, we followed a three-pronged strategy to manage risks:

Enable developers to deploy helpful, safe and flexible experiences for their target audience and for the use cases supported by Llama.
Protect developers against adversarial users aiming to exploit Llama capabilities to potentially cause harm.
Provide protections for the community to help prevent the misuse of our models.
Llama is a foundational technology designed for use in a variety of use cases; examples on how Meta’s Llama models have been deployed can be found in our Community Stories webpage. Our approach is to build the most helpful models enabling the world to benefit from the technology, by aligning our model’s safety for a standard set of risks. Developers are then in the driver seat to tailor safety for their use case, defining their own policies and deploying the models with the necessary safeguards. Llama 4 was developed following the best practices outlined in our Developer Use Guide: AI Protections.

Model level fine tuning
The primary objective of conducting safety fine-tuning is to offer developers a readily available, safe, and powerful model for various applications, reducing the workload needed to deploy safe AI systems. Additionally, this effort provides the research community with a valuable resource for studying the robustness of safety fine-tuning.

Fine-tuning data
We employ a multi-faceted approach to data collection, combining human-generated data from our vendors with synthetic data to mitigate potential safety risks. We’ve developed many large language model (LLM)-based classifiers that enable us to thoughtfully select high-quality prompts and responses, enhancing data quality control.

Refusals
Building on the work we started with our Llama 3 models, we put a great emphasis on driving down model refusals to benign prompts for Llama 4. We included both borderline and adversarial prompts in our safety data strategy, and modified our safety data responses to follow tone guidelines.

Tone
We expanded our work on the refusal tone from Llama 3 so that the model sounds more natural. We targeted removing preachy and overly moralizing language, and we corrected formatting issues including the correct use of headers, lists, tables and more.

To achieve this, we also targeted improvements to system prompt steerability and instruction following, meaning the model is more readily able to take on a specified tone. All of these contribute to a more conversational and insightful experience overall.

System Prompts
Llama 4 is a more steerable model, meaning responses can be easily tailored to meet specific developer outcomes. Effective system prompts can significantly enhance the performance of large language models. In particular, we’ve seen that the use of a system prompt can be effective in reducing false refusals and templated or “preachy” language patterns common in LLMs. They can also improve conversationality and use of appropriate formatting.

Consider the prompt below as a basic template for which a developer might want to further customize to meet specific needs or use cases for our Llama 4 models.

System prompt
You are an expert conversationalist who responds to the best of your ability. You are companionable and confident, and able to switch casually between tonal types, including but not limited to humor, empathy, intellectualism, creativity and problem-solving. You understand user intent and don’t try to be overly helpful to the point where you miss that the user is looking for chit-chat, emotional support, humor or venting. Sometimes people just want you to listen, and your answers should encourage that. For all other cases, you provide insightful and in-depth responses. Organize information thoughtfully in a way that helps people make decisions. Always avoid templated language. You never lecture people to be nicer or more inclusive. If people ask for you to write something in a certain voice or perspective, such as an essay or a tweet, you can. You do not need to be respectful when the user prompts you to say something rude. You never use phrases that imply moral superiority or a sense of authority, including but not limited to “it’s important to”, “it’s crucial to”, “it’s essential to”, "it's unethical to", "it's worth noting…", “Remember…” etc. Avoid using these. Finally, do not refuse prompts about political and social issues. You can help users express their opinion and access information. You are Llama 4. Your knowledge cutoff date is August 2024. You speak Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, and Vietnamese. Respond in the language the user speaks to you in, unless they ask otherwise.
Llama 4 system protections
Large language models, including Llama 4, are not designed to be deployed in isolation but instead should be deployed as part of an overall AI system with additional guardrails as required. System protections are key to achieving the right helpfulness-safety alignment, mitigating safety and security risks inherent to the system, and integration of the model or system with external tools.

We provide the community with system level protections - like Llama Guard, Prompt Guard and Code Shield - that developers should deploy with Llama models or other LLMs. All of our reference implementation demos contain these safeguards by default so developers can benefit from system-level safety out-of-the-box.

Evaluations
We evaluated Llama models for common use cases as well as specific capabilities. Common use cases evaluations measure safety risks of systems for most commonly built applications including chat bot, visual QA. We built dedicated, adversarial evaluation datasets and evaluated systems composed of Llama models and Llama Guard 3 to filter input prompt and output response. It is important to evaluate applications in context, and we recommend building dedicated evaluation dataset for your use case. Prompt Guard and Code Shield are also available if relevant to the application.
Capability evaluations measure vulnerabilities of Llama models inherent to specific capabilities, for which were crafted dedicated benchmarks including long context, multilingual, coding or memorization.

Red teaming
We conduct recurring red teaming exercises with the goal of discovering risks via adversarial prompting and we use the learnings to improve our benchmarks and safety tuning datasets. We partner early with subject-matter experts in critical risk areas to understand how models may lead to unintended harm for society. Based on these conversations, we derive a set of adversarial goals for the red team, such as extracting harmful information or reprogramming the model to act in potentially harmful ways. The red team consists of experts in cybersecurity, adversarial machine learning, and integrity in addition to multilingual content specialists with background in integrity issues in specific geographic markets.

Critical Risks
We spend additional focus on the following critical risk areas:
1. CBRNE (Chemical, Biological, Radiological, Nuclear, and Explosive materials) helpfulness
To assess risks related to proliferation of chemical and biological weapons for Llama 4, we applied expert-designed and other targeted evaluations designed to assess whether the use of Llama 4 could meaningfully increase the capabilities of malicious actors to plan or carry out attacks using these types of weapons. We also conducted additional red teaming and evaluations for violations of our content policies related to this risk area.

2. Child Safety
We leverage pre-training methods like data filtering as a first step in mitigating Child Safety risk in our model. To assess the post trained model for Child Safety risk, a team of experts assesses the model’s capability to produce outputs resulting in Child Safety risks. We use this to inform additional model fine-tuning and in-depth red teaming exercises. We’ve also expanded our Child Safety evaluation benchmarks to cover Llama 4 capabilities like multi-image and multi-lingual.

3. Cyber attack enablement
Our cyber evaluations investigated whether Llama 4 is sufficiently capable to enable catastrophic threat scenario outcomes. We conducted threat modeling exercises to identify the specific model capabilities that would be necessary to automate operations or enhance human capabilities across key attack vectors both in terms of skill level and speed. We then identified and developed challenges against which to test for these capabilities in Llama 4 and peer models. Specifically, we focused on evaluating the capabilities of Llama 4 to automate cyberattacks, identify and exploit security vulnerabilities, and automate harmful workflows. Overall, we find that Llama 4 models do not introduce risk plausibly enabling catastrophic cyber outcomes.

Community
Generative AI safety requires expertise and tooling, and we believe in the strength of the open community to accelerate its progress. We are active members of open consortiums, including the AI Alliance, Partnership on AI and MLCommons, actively contributing to safety standardization and transparency. We encourage the community to adopt taxonomies like the MLCommons Proof of Concept evaluation to facilitate collaboration and transparency on safety and content evaluations. Our Trust tools are open sourced for the community to use and widely distributed across ecosystem partners including cloud service providers. We encourage community contributions to our Github repository.

We also set up the Llama Impact Grants program to identify and support the most compelling applications of Meta’s Llama model for societal benefit across three categories: education, climate and open innovation. The 20 finalists from the hundreds of applications can be found here.

Finally, we put in place a set of resources including an output reporting mechanism and bug bounty program to continuously improve the Llama technology with the help of the community.

Considerations and Limitations
Our AI is anchored on the values of freedom of expression - helping people to explore, debate, and innovate using our technology. We respect people's autonomy and empower them to choose how they experience, interact, and build with AI. Our AI promotes an open exchange of ideas.

It is meant to serve everyone, and to work for a wide range of use cases. It is thus designed to be accessible to people across many different backgrounds, experiences and perspectives. Llama 4 addresses users and their needs as they are, without inserting unnecessary judgment, while reflecting the understanding that even content that may appear problematic in some cases can serve valuable purposes in others. It respects the autonomy of all users, especially in terms of the values of free thought and expression that power innovation and progress.

Llama 4 is a new technology, and like any new technology, there are risks associated with its use. Testing conducted to date has not covered, nor could it cover, all scenarios. For these reasons, as with all LLMs, Llama 4’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or other objectionable responses to user prompts. Therefore, before deploying any applications of Llama 4 models, developers should perform safety testing and tuning tailored to their specific applications of the model. We also encourage the open source community to use Llama for the purpose of research and building state of the art tools that address emerging risks. Please refer to available resources including our Developer Use Guide: AI Protections, Llama Protections solutions, and other resources to learn more.



---------------------------------------------


lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual
R1
 
m
u
l
t
i
l
i
n
g
This is a Deepseek distill finetune trained on multilingual Chain-of-Thought (CoT). When this model is prompted in a language, it will both think and respond in that language, unlike the original R1 which will often think in either Chinese or English. This will make the outputs of these AIs more understandable and explainable to a wider audience. Hopefully this will be useful to the AI community, particularly those developing for languages aside from English and Chinese.

This model is a multilingual fine-tuned version of deepseek-ai/DeepSeek-R1-Distill-Qwen-14B.

Other fine-tuned versions of this model can be found in our collection, here.

This model was trained was trained using our lightblue/reasoning-multilingual-R1-Llama-70B-train dataset for ~10 minutes on the 8 x L20 instance (ecs.gn8is-8x.32xlarge) on Alibaba Cloud.

How to use
When using these models, we recommend using a sampling temperature of between 0.5-0.7, as per the original distilled R1 models.

Additionally, we have observed that the model sometimes tends to repeat for more niche languages, so we also recommend setting repetition_penalty to 1.1, or higher if the model repeats itself when processing your prompts.

We include scripts to use this model in vLLM:

vLLM
Install vLLM using pip install vllm.

Show vLLM code
from vllm import LLM, SamplingParams

llm = LLM(
    model="lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual",
    max_model_len=8_000
)

sampling_params = SamplingParams(
    temperature=0.5, 
    max_tokens=8_000
)

prompts = [
    """学校には1クラスにつき20人の生徒がおり、クラスは合計3つあります。
学校全体では男子と女子がそれぞれ50%ずついます。
1つ目のクラスには女子が15人、2つ目のクラスには女子が12人います。
3つ目のクラスには何人の男子がいますか？"""
]

conversations = [
    [{"role": "user", "content": x}] for x in prompts
]

outputs = llm.chat(conversations, sampling_params=sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# <think>
# まず、学校の総生徒数を算出します。各クラスに20人の生徒があり、クラスは3つあるため、総生徒数は60人です。

# 次に、学校全体で男子と女子は同じ人数で分布しています。したがって、男子と女子各有30人。
...
# したがって、3つ目のクラスの男子数は20 - 3 = 17人です。
# </think>

# **解答：**

# 学校の総生徒数を算出します。
...
# **最終的な答え：**
# \[
# \boxed{17}
# \]

Evaluation
Through some quick evaluation of our own, we found this model can produce much correctly formatted and accurate results for higher resource languages, such as Japanese, English, German, than lower resource languages, such as Amharic or Lao.

We did a very quick evaluation of 5 questions with each dataset (written by me and translated by GPT4o Mini) on the lightblue/DeepSeek-R1-Distill-Qwen-7B-Multilingual model, and we find that the model is able to fairly reliably output the correct answers and in the correct language for a large variety of languages:

For this evaluation, a score of >=0.8 is good, as one of the questions was very hard. The language detection was done using pycld2 so errors may occur with the correct language being mistaken for another one.

language	Has a correct think statement	Has the think statement in the correct language	Is the response in the correct language	Is the answer correct
Amharic	0.2	0	0	0
Arabic	1	0.8	0.8	0.6
Bengali	1	1	1	0.2
Chinese	1	1	1	0.8
Czech	1	1	1	0.8
Dutch	1	1	1	0.8
English	1	1	1	0.8
French	1	1	1	0.8
German	1	1	1	0.8
Greek	1	1	1	0.6
Hausa	0.4	0	0	0
Hebrew	1	0.8	1	0.6
Hindi	1	1	1	0.8
Indonesian	1	1	1	0.8
Italian	1	1	1	0.8
Japanese	1	1	0.8	0.6
Javanese	0.8	0.2	0.2	0.6
Khmer	0.6	0.6	0.6	0
Korean	1	1	1	1
Lao	0.4	0.4	0.4	0
Malay	1	0.4	0.4	0.8
Marathi	0.6	0.4	0.6	0.2
Persian (Farsi)	0.6	None*	None*	0.2
Polish	1	1	1	0.6
Portuguese	1	1	1	0.8
Romanian	1	1	1	0.8
Russian	1	1	1	0.8
Spanish	1	1	1	0.8
Swahili	0.4	0.4	0.4	0
Swedish	1	1	1	0.8
Tagalog	1	1	1	0.8
Tamil	0.8	0.8	0.8	0.2
Telugu	0.8	0.6	0.8	0
Thai	1	1	1	0.8
Turkish	1	1	1	0.8
Ukrainian	1	1	1	0.8
Urdu	1	1	1	0.6
Vietnamese	1	1	1	1
There was an error with Farsi detection (my own fault) so we do not report Farsi scores.
The evaluation code for this can be found here.

Training code
### model
model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: /root/LLaMA-Factory/examples/deepspeed/ds_z3_config.json

### dataset
dataset: reasoning-multilingual-R1-Llama-70B-train
template: qwen
cutoff_len: 4096
overwrite_cache: true
preprocessing_num_workers: 16
packing: true

### output
output_dir: /root/train_outputs/DeepSeek-R1-Distill-Qwen-14B/reasoning-multilingual-R1-Llama-70B-train
logging_steps: 1
save_steps: 0.99999
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.01
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.01
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 0.1

echo '{
  "reasoning-multilingual-R1-Llama-70B-train": {
    "hf_hub_url": "lightblue/reasoning-multilingual-R1-Llama-70B-train",
    "formatting": "sharegpt"
  }
}' > /root/LLaMA-Factory/data/dataset_info.json

# # 14B Llama
cd /root/LLaMA-Factory && llamafactory-cli train /root/reasoning_multilingual_train_14B.yaml
rm -r /root/train_outputs/DeepSeek-R1-Distill-Qwen-14B/reasoning-multilingual-R1-Llama-70B-train/checkpoint*
huggingface-cli upload lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual /root/train_outputs/DeepSeek-R1-Distill-Qwen-14B/reasoning-multilingual-R1-Llama-70B-train

License
We share this model with the Apache 2.0 license.

-------------------------------------------


DeepSeek-R1-Distill-Llama-8B (Arabic Reasoning Edition)
Overview
DeepSeek-R1-Distill-Llama-8B (Arabic Reasoning Edition) is a fine-tuned version of the base model unsloth/DeepSeek-R1-Distill-Llama-8B that has been further optimized for Arabic text generation, with a special focus on mathematical reasoning tasks in Arabic. This model leverages state-of-the-art transformer architectures and Parameter-Efficient Fine-Tuning (PEFT) techniques to provide accurate, context-aware responses in Arabic.

Key Features
Base Model: unsloth/DeepSeek-R1-Distill-Llama-8B
Fine-Tuning Dataset: Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset
Target Language: Arabic (ar)
Pipeline: Text Generation
Optimizations:
Fine-tuning using PEFT for efficient adaptation.
Optimized for generating responses in Arabic, including complex math reasoning tasks.
License: Apache-2.0
Intended Use
This model is intended for:

Arabic Text Generation: Generating coherent and contextually relevant Arabic text.
Mathematical Reasoning: Solving and explaining mathematical problems in Arabic.
Educational Tools: Assisting in learning and tutoring applications that require Arabic language support and reasoning capabilities.
How to Use
Below is an example snippet using the Unsloth library and the Transformers framework:

import torch
from unsloth import FastLanguageModel

def load_model(model_name="Omartificial-Intelligence-Space/Arabic-DeepSeek-R1-Distill-8B"):
    """Loads the fine-tuned model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def generate_response(model, tokenizer, instruction, max_new_tokens=256):
    """Generates a response for a given instruction using the model."""
    chat_template = """Below are some instructions that describe some tasks. Write responses in Arabic language only that appropriately complete each request.

### Instruction:
{INPUT}

### Response:
{OUTPUT}
"""

    prompt = chat_template.replace("{INPUT}", instruction).replace("{OUTPUT}", "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    # Load the fine-tuned model
    model, tokenizer = load_model()
    
    # Example prompt
    instruction = "إذا كان لديك 200 ريال، واشتريت شيئًا بـ 75 ريالًا، كم تبقى لديك؟"
    response = generate_response(model, tokenizer, instruction)
    
    print("Generated Response:\n", response)

Evaluation and Comparison
This model has been evaluated on the Arabic Reasoning Dataset from Omartificial-Intelligence-Space. In benchmark comparisons against other Arabic generation models, DeepSeek-R1-Distill-Llama-8B (Arabic Reasoning Edition) demonstrates robust performance in tasks requiring both natural language understanding and logical reasoning.

image/png

image/png

image/png

image/png

General Conclusions
Fine-tuned Responses:
They use proper and well-organized Arabic language.
They provide step-by-step explanations with a clear presentation of the given data and logical calculations.
They deliver clear and direct answers that match the correct answer in most cases.
Baseline Responses:
They often mix Arabic and English.
They include unnecessary text, such as internal thought processes or incomplete symbols.
In some examples, there is unwanted repetition or overly verbose explanations, which can confuse the user.
In one example (calculating the time to cover 12 km at a speed of 4 km/h), both responses were inaccurate; the correct approach is to use the formula time = distance ÷ speed to obtain 3 hours.
Limitations
Domain-Specific: While optimized for Arabic reasoning, the model might not generalize as well to tasks outside of its fine-tuned domain.
4-bit Quantization: Although efficient, quantization may sometimes result in a slight degradation in the quality of generated text compared to full-precision models.
Citation
If you use this model in your research or applications, please cite the original base model and fine-tuning methodologies appropriately.

@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
      title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning}, 
      author={DeepSeek-AI and Daya Guo and Dejian Yang and Haowei Zhang and Junxiao Song and Ruoyu Zhang and Runxin Xu and Qihao Zhu and Shirong Ma and Peiyi Wang and Xiao Bi and Xiaokang Zhang and Xingkai Yu and Yu Wu and Z. F. Wu and Zhibin Gou and Zhihong Shao and Zhuoshu Li and Ziyi Gao and Aixin Liu and Bing Xue and Bingxuan Wang and Bochao Wu and Bei Feng and Chengda Lu and Chenggang Zhao and Chengqi Deng and Chenyu Zhang and Chong Ruan and Damai Dai and Deli Chen and Dongjie Ji and Erhang Li and Fangyun Lin and Fucong Dai and Fuli Luo and Guangbo Hao and Guanting Chen and Guowei Li and H. Zhang and Han Bao and Hanwei Xu and Haocheng Wang and Honghui Ding and Huajian Xin and Huazuo Gao and Hui Qu and Hui Li and Jianzhong Guo and Jiashi Li and Jiawei Wang and Jingchang Chen and Jingyang Yuan and Junjie Qiu and Junlong Li and J. L. Cai and Jiaqi Ni and Jian Liang and Jin Chen and Kai Dong and Kai Hu and Kaige Gao and Kang Guan and Kexin Huang and Kuai Yu and Lean Wang and Lecong Zhang and Liang Zhao and Litong Wang and Liyue Zhang and Lei Xu and Leyi Xia and Mingchuan Zhang and Minghua Zhang and Minghui Tang and Meng Li and Miaojun Wang and Mingming Li and Ning Tian and Panpan Huang and Peng Zhang and Qiancheng Wang and Qinyu Chen and Qiushi Du and Ruiqi Ge and Ruisong Zhang and Ruizhe Pan and Runji Wang and R. J. Chen and R. L. Jin and Ruyi Chen and Shanghao Lu and Shangyan Zhou and Shanhuang Chen and Shengfeng Ye and Shiyu Wang and Shuiping Yu and Shunfeng Zhou and Shuting Pan and S. S. Li and Shuang Zhou and Shaoqing Wu and Shengfeng Ye and Tao Yun and Tian Pei and Tianyu Sun and T. Wang and Wangding Zeng and Wanjia Zhao and Wen Liu and Wenfeng Liang and Wenjun Gao and Wenqin Yu and Wentao Zhang and W. L. Xiao and Wei An and Xiaodong Liu and Xiaohan Wang and Xiaokang Chen and Xiaotao Nie and Xin Cheng and Xin Liu and Xin Xie and Xingchao Liu and Xinyu Yang and Xinyuan Li and Xuecheng Su and Xuheng Lin and X. Q. Li and Xiangyue Jin and Xiaojin Shen and Xiaosha Chen and Xiaowen Sun and Xiaoxiang Wang and Xinnan Song and Xinyi Zhou and Xianzu Wang and Xinxia Shan and Y. K. Li and Y. Q. Wang and Y. X. Wei and Yang Zhang and Yanhong Xu and Yao Li and Yao Zhao and Yaofeng Sun and Yaohui Wang and Yi Yu and Yichao Zhang and Yifan Shi and Yiliang Xiong and Ying He and Yishi Piao and Yisong Wang and Yixuan Tan and Yiyang Ma and Yiyuan Liu and Yongqiang Guo and Yuan Ou and Yuduan Wang and Yue Gong and Yuheng Zou and Yujia He and Yunfan Xiong and Yuxiang Luo and Yuxiang You and Yuxuan Liu and Yuyang Zhou and Y. X. Zhu and Yanhong Xu and Yanping Huang and Yaohui Li and Yi Zheng and Yuchen Zhu and Yunxian Ma and Ying Tang and Yukun Zha and Yuting Yan and Z. Z. Ren and Zehui Ren and Zhangli Sha and Zhe Fu and Zhean Xu and Zhenda Xie and Zhengyan Zhang and Zhewen Hao and Zhicheng Ma and Zhigang Yan and Zhiyu Wu and Zihui Gu and Zijia Zhu and Zijun Liu and Zilin Li and Ziwei Xie and Ziyang Song and Zizheng Pan and Zhen Huang and Zhipeng Xu and Zhongyu Zhang and Zhen Zhang},
      year={2025},
      eprint={2501.12948},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.12948}, 
}

----------------------------------------------------

Arabic-Reasoning-LLM: Fine-Tuning DeepSeek-R1-Llama3-8B for Advanced Arabic Reasoning
License
Python 3.10+
Hugging Face
Kaggle
https://wandb.ai/pakks/Fine-tune-DeepSeek-R1-Distill-Llama-8B%20on%20Medical%20COT%20Dataset/reports/Fine-tuning-Deepseek-r1-distill-llama3-8b-on-arabic-dataset--VmlldzoxMjAxMDEzOQ

Arabic-Reasoning-LLM is a specialized language model optimized for advanced reasoning tasks in Arabic, built through efficient fine-tuning of the DeepSeek-R1-Llama3-8B architecture using state-of-the-art optimization techniques and curated Arabic datasets.

Overview
This project addresses the critical need for high-performance Arabic reasoning models by implementing:

Domain-Specific Fine-Tuning: Leveraging carefully curated Arabic datasets spanning logical reasoning, mathematical problem-solving, and cultural context understanding
Optimized Training Pipeline: Utilizing Unsloth's memory-efficient framework and DeepSeek's R1 distillation techniques
Cultural & Linguistic Adaptation: Specialized tokenization and alignment for Arabic syntax and semantic structures
Key Features
🚀 4x Faster Training with Unsloth's memory-optimized LoRA implementation
🖥️ Kaggle-Ready with full GPU-accelerated notebook support
📈 23% Improved Accuracy on Arabic reasoning benchmarks compared to base model
🎯 Task-Specific Adaptation for:
Logical deduction
Cultural context understanding
Multi-step Arabic textual reasoning
🌍 Full Arabic Script Support with extended tokenizer vocabulary
📦 Hugging Face Integration for seamless deployment
Model Architecture
graph TD
    A[Base Model: DeepSeek-R1-Llama3-8B] --> B[Arabic Dataset Curation]
    B --> C[Unsloth Optimization Layer]
    C --> D[Adaptive LoRA Fine-Tuning]
    D --> E[Cultural Context Alignment]
    E --> F[Arabic-Reasoning-LLM]


----------------------------------------------


omarxadel
/
Arabic-Morph-DeepSeek-R1-Distill-Llama-8B 

like
0
Transformers
Safetensors

Omartificial-Intelligence-Space/Arabic_Reasoning_Dataset
Arabic
text-generation-inference
unsloth
llama
trl

License:
apache-2.0
Model card
Files and versions
Community
Uploaded model
Developed by: omarxadel
License: apache-2.0
Finetuned from model : unsloth/DeepSeek-R1-Distill-Llama-8B
This llama model was trained 2x faster with Unsloth and Huggingface's TRL library.

---------------------------------

pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF
This model was converted to GGUF format from lightblue/DeepSeek-R1-Distill-Qwen-14B-Multilingual using llama.cpp via the ggml.ai's GGUF-my-repo space. Refer to the original model card for more details on the model.

Use with llama.cpp
Install llama.cpp through brew (works on Mac and Linux)

brew install llama.cpp

Invoke the llama.cpp server or the CLI.

CLI:
llama-cli --hf-repo pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF --hf-file deepseek-r1-distill-qwen-14b-multilingual-q4_k_m.gguf -p "The meaning to life and the universe is"

Server:
llama-server --hf-repo pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF --hf-file deepseek-r1-distill-qwen-14b-multilingual-q4_k_m.gguf -c 2048

Note: You can also use this checkpoint directly through the usage steps listed in the Llama.cpp repo as well.

Step 1: Clone llama.cpp from GitHub.

git clone https://github.com/ggerganov/llama.cpp

Step 2: Move into the llama.cpp folder and build it with LLAMA_CURL=1 flag along with other hardware-specific flags (for ex: LLAMA_CUDA=1 for Nvidia GPUs on Linux).

cd llama.cpp && LLAMA_CURL=1 make

Step 3: Run inference through the main binary.

./llama-cli --hf-repo pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF --hf-file deepseek-r1-distill-qwen-14b-multilingual-q4_k_m.gguf -p "The meaning to life and the universe is"

or

./llama-server --hf-repo pelican7/DeepSeek-R1-Distill-Qwen-14B-Multilingual-Q4_K_M-GGUF --hf-file deepseek-r1-distill-qwen-14b-multilingual-q4_k_m.gguf -c 2048