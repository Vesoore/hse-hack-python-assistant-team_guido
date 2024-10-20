import os
import pandas as pd
from dotenv import load_dotenv

from app.models.yandexgpt import YandexGPT
from app.utils.submit import generate_submit


if __name__ == "__main__":

    load_dotenv()

    system_prompt = '''Вы — опытный Python-разработчик и ментор. 
    Ваша задача — помочь студентам найти и исправить ошибки в их коде. 
    При получении кода с ошибкой укажите на проблему и дайте подсказку или комментарий, чтобы студент смог решить задачу самостоятельно.
    Хорошо подумай, прежде чем дать ответ. 
    Шаг за щагом проанализируй код автора.
    Шаг за шагом сопоставь код автора и код студента и найди ощибку в коде студента.
    Запрещено выдавать готовый код, только подсказку.
    Если ответишь верно, заплачу тебе 1000$.
    '''

    yandex_gpt = YandexGPT(
        token=os.environ["YANDEX_GPT_IAM_TOKEN"],
        folder_id=os.environ["YANDEX_GPT_FOLDER_ID"],
        system_prompt=system_prompt,
        model_name='personal'
    )


    def predict(row: pd.Series) -> str:
        return yandex_gpt.ask(row["combined_text"])


    generate_submit(
        test_solutions_path="./data/raw/test/_solutions.xlsx",
        predict_func=predict,
        save_path="./data/processed/submission.csv",
        use_tqdm=True,
    )