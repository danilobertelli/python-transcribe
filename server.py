import os
import tempfile
import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from transformers import pipeline

# Configura o logging para o nível INFO
logging.basicConfig(level=logging.INFO)

# Carrega o pipeline globalmente para evitar carregamento a cada requisição
logging.info("Carregando o pipeline de ASR...")
asr_pipeline = pipeline("automatic-speech-recognition", model="nilc-nlp/distil-whisper-coraa-mupe-asr")
logging.info("Pipeline carregado com sucesso.")

app = FastAPI()

# Configuração de CORS (ajuste os domínios conforme necessário)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Endpoint que recebe um arquivo WAV e retorna a transcrição.
    """
    logging.info(f"Recebendo arquivo: {file.filename}")
    try:
        # Lê os bytes do arquivo recebido
        contents = await file.read()
        logging.info("Arquivo lido com sucesso.")

        # Cria um arquivo temporário e fecha-o antes de usar
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp.flush()  # Garante que os dados estejam gravados
            temp_file_path = tmp.name
        logging.info(f"Arquivo temporário criado em {temp_file_path}")

        # Executa a transcrição usando o pipeline
        logging.info("Iniciando transcrição...")
        transcription = asr_pipeline(temp_file_path)
        logging.info("Transcrição concluída com sucesso.")

        # Remove o arquivo temporário
        os.remove(temp_file_path)
        logging.info("Arquivo temporário removido.")

        # Retorna a transcrição (geralmente um dicionário com a chave 'text')
        return transcription

    except Exception as e:
        logging.error("Erro ao processar o arquivo", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    logging.info("Iniciando o servidor na porta 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
