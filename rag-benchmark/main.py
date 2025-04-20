import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from datasets import Dataset
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from settings import settings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from huggingface_hub import login

login(token=settings.get_api_token())
# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Evaluation Service",
    description="Сервис для оценки качества RAG систем",
    version="1.0.0"
)

# Модель данных для входного запроса
class RagTestInput(BaseModel):
    questions: List[str]  # Список вопросов
    answers: List[str]  # Список ответов
    contexts: List[List[str]]  # Список контекстов для каждого вопроса
    ground_truths: List[str]  # Список правильных ответов

    class Config:
        json_schema_extra = {
            "example": {
                "questions": ["Какая столица России?"],
                "answers": ["Москва является столицей России"],
                "contexts": [["Москва - столица России", "Город основан в 1147 году"]],
                "ground_truths": ["Москва"]
            }
        }

# Модель данных для ответа
class RagTestResult(BaseModel):
    answer_relevancy: float
    context_precision: float
    context_recall: float
    faithfulness: float

llm = HuggingFaceEndpoint(
    repo_id=settings.get_model_repo_id(),
    task="text-generation",
    max_new_tokens=settings.get_max_new_tokens(),
    temperature=settings.get_temperature(),
    huggingfacehub_api_token=settings.get_api_token()
)

model = ChatHuggingFace(llm=llm)

wrapper = LangchainLLMWrapper(model)

# Создаем эмбеддинги через LangChain
base_embeddings = HuggingFaceEmbeddings(
    model_name=settings.get_embeddings_model_name()
)

# Оборачиваем в RAGAS-совместимый интерфейс
embeddings = LangchainEmbeddingsWrapper(base_embeddings)

@app.post("/evaluate-rag", response_model=RagTestResult)
async def evaluate_rag(input_data: RagTestInput):
    """
    Эндпоинт для оценки RAG системы.
    
    Args:
        input_data (RagTestInput): Входные данные для оценки
        
    Returns:
        RagTestResult: Результаты оценки различных метрик
        
    Raises:
        HTTPException: При ошибках валидации или обработки
    """
    try:
        logger.info(f"Получен запрос на оценку RAG с {len(input_data.questions)} вопросами")
        
        if not (len(input_data.questions) == len(input_data.answers) == 
                len(input_data.contexts) == len(input_data.ground_truths)):
            logger.error("Несоответствие длин входных списков")
            raise HTTPException(
                status_code=400, 
                detail="Все списки должны иметь одинаковую длину"
            )

        data = {
            "question": input_data.questions,
            "answer": input_data.answers,
            "contexts": input_data.contexts,
            "ground_truth": input_data.ground_truths,
        }
        dataset = Dataset.from_dict(data)
        
        metrics = [
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        ]

        logger.info("Начало оценки метрик")
        results = evaluate(
            dataset,
            metrics,
            llm=wrapper,
            embeddings=embeddings,
        )
        logger.info(f"Оценка метрик завершена успешно {results}")

        return RagTestResult(
            answer_relevancy=results["answer_relevancy"],
            context_precision=results["context_precision"],
            context_recall=results["context_recall"],
            faithfulness=results["faithfulness"],
        )

    except Exception as e:
        logger.error(f"Произошла ошибка при обработке запроса: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)