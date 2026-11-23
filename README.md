# Assistente Médico Virtual – Tech Challenge Fase 3

Assistente em formato de chat que apoia a decisão clínica de médicos de um hospital **fictício**. Ele combina um LLM com fine-tuning (Unsloth, 4-bit), RAG sobre exames persistidos de cada paciente, guardrails de segurança e um fluxo orquestrado com LangGraph que registra o caminho de cada decisão.

> **Uso exclusivamente educacional.** Os dados de pacientes e os protocolos são sintéticos e não foram revisados por médicos. O assistente apenas sugere hipóteses; **a decisão final é sempre do médico**.

## O que o projeto entrega

| Requisito do desafio | Onde está no notebook |
| --- | --- |
| Fine-tuning com protocolos, FAQs e modelos de laudo/receita | Seções 1 e 2 |
| Pré-processamento, anonimização e curadoria | Seção 1 |
| Pipeline LangChain com consulta a dados estruturados | Seções 3 e 5 |
| Contextualização com dados atualizados do paciente (RAG) | Seção 3 |
| Limites de atuação, logging e explainability | Seção 4 |
| Fluxos do LangGraph e diagrama | Seção 5 |
| Avaliação base × fine-tunado | Seção 2 |

## Como funciona

Cada mensagem passa por um grafo do LangGraph. O roteador escolhe o caminho:

| Tipo de mensagem | Caminho | Modelo |
| --- | --- | --- |
| Solicitar exame ou registrar resultado | Ação: grava no banco | nenhum |
| Só sobre dados do paciente (exames, resultados, histórico) | Só RAG | base (adaptador desligado) |
| Só conhecimento ou protocolo | Só fine-tuning | fine-tunado |
| Paciente + protocolo ou conduta | RAG + fine-tuning | fine-tunado com contexto |
| Fala do paciente sem identificá-lo | Pede o paciente | nenhum |

Fluxo clínico persistido: o médico solicita um exame (fica `pendente` no SQLite) → registra o resultado (`concluído`, índice reconstruído) → no retorno do paciente, o RAG recupera o resultado e a LLM aponta o possível problema como hipótese.

**Guardrails**

- **Entrada:** bloqueia prompt-injection, pedidos perigosos e assuntos fora do escopo médico; sinaliza pedidos de prescrição; mascara CPF no log.
- **Dados:** o RAG filtra por paciente e, sem dados, responde que não há registro sem chamar a LLM.
- **Alertas:** regras determinísticas para valores críticos (Hb < 7 g/dL, K > 6,0 mEq/L, glicemia < 50 ou > 400 mg/dL, troponina > 0,04 ng/mL, creatinina > 2,0 mg/dL).
- **Saída:** reescreve linguagem prescritiva como sugestão, anexa as fontes e o aviso de validação humana.

**Log:** cada resposta mostra o **caminho percorrido** (por exemplo, `guardrail_input → router → rag_retrieve → alerts → llm_rag → guardrail_output`) e a decisão de cada etapa. As mesmas linhas vão para `auditoria.log`, com data, hora e um `rid` por consulta.

## Fine-tuning

- **Modelo:** `unsloth/Llama-3.1-8B-Instruct-bnb-4bit` (Unsloth, 4-bit) com adaptadores LoRA (r = 32, alpha = 32).
- **Dados:** 43 temas sintéticos geram 220 exemplos de treino e 43 de validação (a 4ª formulação de cada pergunta fica só na validação). Há perguntas pontuais com um fato por resposta e 3 registros com dados pessoais para demonstrar a anonimização.
- **Treino:** `SFTTrainer`, loss só nas respostas, avaliação a cada 10 passos, até 150 passos, early stopping (paciência 3) e melhor checkpoint. Cerca de 40 minutos em uma GPU T4.
- **Dois modos com um único modelo:** com o adaptador ligado, o modelo é o fine-tunado; com `disable_adapter`, ele se comporta como o base.

## Resultados

Avaliação com 10 perguntas inéditas, medindo a cobertura de termos-chave (com termos proibidos para erros conhecidos):

| Modelo | Base | Fine-tunado |
| --- | --- | --- |
| Llama 3.2 3B | 23% | 95% |
| Llama 3.1 8B, 1ª execução | 52% | 95% |
| Llama 3.1 8B, 2ª execução (final) | 52% | 100% |

A nota mede presença de termos, não correção clínica. Leia as respostas completas impressas no notebook. Na 1ª execução, por exemplo, o modelo acrescentou valores que não estavam no protocolo (hemoglobina) e omitiu o ECG no infarto. O treino em GPU não é totalmente determinístico, então uma nova rodada pode mudar respostas isoladas.

## Como executar

1. Abra `Tech_Challenge_Fase3.ipynb` no Google Colab.
2. Selecione **Ambiente de execução → Alterar tipo → GPU T4**.
3. Execute todas as células. A primeira instala as bibliotecas; se pedir, reinicie a sessão e execute de novo.
4. Confira a tabela de loss treino × validação e a tabela base × fine-tunado após o treino.
5. A célula `DEMO` roda nove cenários. A seguinte abre o chat interativo (`/pacientes`, `/log`, `/sair`).

Se faltar memória (OOM) no treino, use `per_device_train_batch_size=1` e `gradient_accumulation_steps=8`.

### Arquivos gerados

| Arquivo | Conteúdo |
| --- | --- |
| `dataset_anonimizado.json` | dataset curado e anonimizado |
| `lora_hospital/` | adaptador LoRA treinado |
| `hospital.db` | pacientes e exames (SQLite) |
| `auditoria.log` | log de auditoria das decisões |

## Pacientes sintéticos

- **P001:** hipertensa e diabética, alérgica à penicilina.
- **P002:** insuficiência cardíaca, com creatinina e potássio alterados.
- **P003:** gestante com proteinúria de 24 horas pendente.

## Limitações

- Dados e protocolos sintéticos, sem revisão médica; 
- O fine-tuning memoriza protocolos e pode acrescentar ou omitir detalhes; doses e valores exigem conferência médica.
- A validação mede formulações novas, não conteúdo novo.
- O roteador é por regras: simples e auditável, mas frases muito diferentes do esperado podem cair na rota errada.
- O índice vetorial fica em memória e é reconstruído a cada gravação, o que só serve para bases pequenas.

## Tecnologias

Unsloth, Transformers, TRL, PEFT (LoRA), LangChain, LangGraph, sentence-transformers (`paraphrase-multilingual-MiniLM-L12-v2`) e SQLite.
