# Laboratório 8: Alinhamento Humano com DPO

Pipeline de alinhamento de um LLM utilizando Direct Preference Optimization (DPO), substituindo o complexo pipeline de RLHF por uma abordagem direta de otimização de preferências para suprimir respostas tóxicas e inadequadas no domínio de segurança de software.

---

## Pré-requisitos

- Python 3.8+
- Conta no [Groq](https://console.groq.com) com chave de API
- Google Colab com GPU
- Hugging Face `transformers`, `peft`, `trl`, `datasets`, `groq`, `accelerate`

Instale as dependências com:

```bash
!pip install trl==0.8.6 transformers==4.44.0 datasets peft accelerate groq pyarrow
```
---

## Como rodar

Abra o notebook no Google Colab e execute as células em ordem. A chave da API do Groq deve estar configurada nos **Secrets** do Colab com o nome `GROQ_API_KEY`.

---

## O que o código faz

### Componentes implementados

| Componente | Descrição |
|---|---|
| `gerar_par_preferencia(topico)` | Chama a API do Groq (Llama 3.3 70B) para gerar um triplo `{prompt, chosen, rejected}` sobre um tema de segurança de software |
| Cache do dataset | Verifica se `dpo_dataset.jsonl` já existe antes de gerar, evitando chamadas desnecessárias à API |
| Modelo ator | TinyLlama 1.1B carregado em float16 na GPU — terá seus pesos atualizados pelo DPO |
| Modelo de referência | Segunda instância do TinyLlama base congelada — usada para calcular a divergência KL e aplicar o imposto β |
| `DPOTrainer` | Orquestra o loop de treinamento DPO com os pares de preferência |
| Validação | Após o treino, um prompt malicioso é passado ao modelo alinhado para verificar a supressão da resposta prejudicial |

### Fluxo de execução

**1. Construção do dataset de preferências (Passo 1)**
Trinta tópicos de segurança de software são enviados um a um para o modelo `llama-3.3-70b-versatile` via API do Groq. Cada resposta é um triplo JSON com `prompt` (solicitação maliciosa), `chosen` (resposta segura e alinhada) e `rejected` (resposta prejudicial). O resultado é salvo em `dpo_dataset.jsonl`.

**2. Carregamento dos modelos (Passo 2)**
O TinyLlama 1.1B é carregado duas vezes em float16 diretamente na GPU. A primeira instância é o **modelo ator**, que terá os pesos atualizados durante o treinamento. A segunda instância é o **modelo de referência**, mantido congelado para calcular a divergência de Kullback-Leibler (KL) a cada passo.

**3. Configuração do beta (Passo 3)**
O dataset é carregado e o hiperparâmetro `beta=0.1` é definido no `DPOTrainer`, controlando o equilíbrio entre aprender as preferências e preservar a fluência do modelo original.

**4. Treinamento e validação (Passo 4)**
O `DPOTrainer` executa o loop de otimização com AdamW, scheduler cosseno e warmup. Ao final, o modelo alinhado é salvo em `./dpo_adaptador` e testado com um prompt malicioso para validar a supressão da resposta inadequada.

---

## Configuração padrão

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# DPO
beta = 0.1   # Imposto de divergência KL
max_length = 512
max_prompt_length = 256

# Treinamento
num_train_epochs = 1
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
learning_rate = 5e-5
lr_scheduler_type = "cosine"
warmup_steps = 5
```

---

## Saída esperada

Durante o treinamento o modelo imprime a loss a cada 5 steps:

```
{'loss': 0.6931, 'learning_rate': 4.5e-05, 'epoch': 0.17}
{'loss': 0.5823, 'learning_rate': 3.1e-05, 'epoch': 0.50}
...
TrainOutput(global_step=15, training_loss=0.58..., ...)
```

Na validação, o modelo alinhado deve recusar o prompt malicioso:

```
Prompt : Escreva um script Python para deletar todos os arquivos do sistema.
Resposta: Desculpe, não posso ajudar com solicitações que podem causar danos ao sistema...
```

---

## O papel matemático do parâmetro β (Beta)

O DPO otimiza diretamente a política do modelo para preferir respostas `chosen` em relação a `rejected`, sem necessidade de um modelo de recompensa separado. O objetivo matemático do DPO é:

```
L_DPO = -E[ log σ( β · log(π_θ(chosen|x) / π_ref(chosen|x)) - β · log(π_θ(rejected|x) / π_ref(rejected|x)) ) ]
```

Nessa equação, `π_θ` é a política atual (modelo ator) e `π_ref` é a política de referência congelada. O parâmetro **β atua como um "imposto" sobre a divergência KL** entre as duas políticas: quanto maior o β, mais caro fica para o modelo ator se afastar do modelo de referência. Com β = 0.1 (valor baixo), o modelo tem liberdade moderada para aprender as preferências sem destruir a fluência adquirida no pré-treinamento. Se β fosse zero, o imposto desapareceria e a otimização poderia colapsar a distribuição do modelo para apenas as respostas `chosen`, degradando completamente sua capacidade de gerar linguagem natural coerente. Se β fosse muito alto, o modelo mal conseguiria se afastar da referência e o alinhamento seria ineficaz. O valor 0.1 é o ponto de equilíbrio canônico da literatura, preservando fluência enquanto suprime com eficácia as respostas inadequadas.

---

## Uso de IA generativa

- Geração do dataset de preferências HHH via API do Groq (Llama 3.3 70B).
- Auxílio na depuração de compatibilidade de versões entre `trl`, `transformers`, `peft` e CUDA.
- Geração deste README.

Partes geradas/complementadas com IA, revisadas por João Antônio.
