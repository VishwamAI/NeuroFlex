import torch
from generative_ai_model import T5Model, DevInAI, ChainOfThought, MultiPromptHandler

def evaluate_model(devin_ai, chain_of_thought, multi_prompt_handler):
    print("Evaluating Generative AI Text-to-Text Model")
    print("===========================================")

    # Test basic text generation
    prompt = "Translate the following English text to French: 'Hello, how are you?'"
    response = devin_ai.generate_multiple_texts([prompt])[0]
    print(f"\nBasic Text Generation:")
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")

    # Test chain-of-thought reasoning
    initial_prompt = "Explain the process of photosynthesis"
    reasoning_steps = chain_of_thought.reason_through_steps(initial_prompt, num_steps=3)
    print("\nChain-of-Thought Reasoning:")
    for i, step in enumerate(reasoning_steps):
        print(f"Step {i}: {step}")

    # Test multi-prompt development
    prompts = [
        "What is the capital of France?",
        "Explain quantum physics in simple terms.",
        "Describe the water cycle."
    ]
    multi_prompt_response = multi_prompt_handler.generate_with_multi_prompt(prompts)
    print("\nMulti-Prompt Development:")
    print(f"Prompts: {prompts}")
    print(f"Combined Response: {multi_prompt_response}")

if __name__ == "__main__":
    # Initialize model and components
    model = T5Model()
    devin_ai = DevInAI(model)
    chain_of_thought = ChainOfThought(devin_ai)
    multi_prompt_handler = MultiPromptHandler(devin_ai)

    # Evaluate the model
    evaluate_model(devin_ai, chain_of_thought, multi_prompt_handler)

    print("\nEvaluation complete. Please review the outputs to assess the model's performance.")
