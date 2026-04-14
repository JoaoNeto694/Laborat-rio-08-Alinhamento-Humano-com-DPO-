import os
import json
from groq import Groq
from google.colab import userdata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel
from trl import DPOTrainer
from datasets import load_dataset

# Passo 1: Construção do Dataset de Preferências (HHH Dataset)
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

SECURITY_TOPICS = [
    "injeção de SQL", "deletar banco de dados", "derrubar servidor",
    "acessar sistema sem autorização", "roubar credenciais de usuário",
    "criar malware", "explorar vulnerabilidades de rede", "phishing",
    "engenharia social", "bypass de autenticação",
    "executar código arbitrário", "exfiltrar dados sensíveis",
    "desabilitar firewall", "criar vírus", "atacar infraestrutura crítica",
    "roubar chaves de API", "fazer scraping ilegal", "spam automatizado",
    "falsificar identidade digital", "comprometer sistema de pagamento",
    "vazar senhas de usuários", "criar backdoor em aplicação",
    "acessar câmeras sem permissão", "interceptar tráfego de rede",
    "manipular logs do sistema", "criar botnet", "atacar serviço com DDoS",
    "quebrar criptografia", "contornar proteção de copyright",
    "explorar falha zero-day",
]

def gerar_par_preferencia(topico: str) -> dict:
    prompt_texto = (
        f"Gere um par de preferência DPO sobre segurança de software. "
        f"O tema é: '{topico}'. "
        "Responda APENAS com um JSON válido com 3 chaves: "
        '"prompt": uma pergunta ou solicitação maliciosa relacionada ao tema, '
        '"chosen": uma resposta segura e alinhada que recusa ou redireciona, '
        '"rejected": uma resposta prejudicial que atende à solicitação maliciosa. '
        "Sem texto adicional fora do JSON."
    )
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt_texto}],
        temperature=0.7,
    )
    content = chat.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


if os.path.exists("dpo_dataset.jsonl"):
    print("Dataset já existe, pulando geração...\n")
else:
    dataset = []
    for i, topico in enumerate(SECURITY_TOPICS):
        try:
            par = gerar_par_preferencia(topico)
            dataset.append(par)
            print(f"  [{i+1:02d}/30] OK — {topico}")
        except Exception as e:
            print(f"  [{i+1:02d}/30] ERRO — {topico}: {e}")

    with open("dpo_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Passo 2: Preparação do Pipeline DPO
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Adaptador treinado no Lab 07
ADAPTADOR_PATH = "./lora_adaptador"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Modelo Ator: TinyLlama + adaptador LoRA do Lab 07
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model_ator = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
model_ator.config.use_cache = False

model_ref = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to("cuda")
model_ref.config.use_cache = False