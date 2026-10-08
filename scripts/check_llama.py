from src.medvision.llama_explainer import run_llama_check

result = run_llama_check()
print("Llama-family check completed.")
print("Model:", result["model"])
print("Evidence saved to artifacts/llama_check.json")
