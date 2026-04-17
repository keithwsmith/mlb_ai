# After training, export merged model to GGUF
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("sqlcoder_mlb_lora")
model.save_pretrained_gguf(
    "sqlcoder_mlb_gguf",
    tokenizer,
    quantization_method = "q4_k_m"   # good balance of size/quality
)
#```
#
# Then create an `Modelfile` for Ollama:
# ```
# FROM ./sqlcoder_mlb_gguf/unsloth.Q4_K_M.gguf
#
# PARAMETER temperature 0
# PARAMETER num_ctx 4096
# PARAMETER num_predict 600
#
# SYSTEM "You are a SQL Server T-SQL expert. Always use TOP(N) not LIMIT. Never use ILIKE, NULLS FIRST/LAST, or :: cast operator."
#
# ollama create sqlcoder-mlb -f Modelfile
# ollama run sqlcoder-mlb
#
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "sqlcoder-mlb")