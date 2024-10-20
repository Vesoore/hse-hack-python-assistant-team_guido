from typing import Callable
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer
from app.utils.toxiclassifier import ToxiClassifier


# Загрузка модели для классификации токсичных комментов
toxic_classifier = ToxiClassifier()

# Загрузка модели и токенайзера
print("Loading models...", end="")
model_name = "DeepPavlov/rubert-base-cased-sentence"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
print("OK")


# Функция для получения эмбеддингов из предложения
def get_sentence_embedding(sentence: str) -> torch.Tensor:
    try:
        # Токенизация и получение эмбеддингов
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        return embedding
    except Exception as e:
        print(f"Ошибка при обработке предложения: {sentence}. Ошибка: {e}")
        return torch.zeros(768)  # Возвращаем пустой тензор в случае ошибки


# Конвертация строки в эмбеддинг
def string2embedding(string: str) -> torch.Tensor:
    return torch.Tensor([float(i) for i in string.split()])


# Конвертация эмбеддинга в строку
def embedding2string(embedding: torch.Tensor) -> str:
    return " ".join([str(i) for i in embedding.tolist()])


# Функция для генерации файла submit
def generate_submit(test_solutions_path: str, predict_func: Callable, save_path: str, use_tqdm: bool = True) -> None:
    # Загрузка данных из Excel
    test_solutions = pd.read_excel(test_solutions_path)

    # Если нужно показывать прогресс, используем tqdm
    bar = range(len(test_solutions))
    if use_tqdm:
        import tqdm
        bar = tqdm.tqdm(bar, desc="Predicting")

    # Создаем DataFrame для сабмита
    submit_df = pd.DataFrame(columns=["solution_id", "author_comment", "author_comment_embedding"])

    # Цикл по каждому решению
    for i in bar:
        try:
            idx = test_solutions.iloc[i]['id']

            solution_row = test_solutions.iloc[i]

            result = toxic_classifier.predict(solution_row['student_solution'])

            if result['label'] == 'OK':

                # Предсказание
                text = predict_func(solution_row)  # здесь можно сделать что угодно с решением студента

                # Получение эмбеддинга и его конвертация в строку
                embedding = embedding2string(get_sentence_embedding(text))

                # Заполнение DataFrame
                submit_df.loc[i] = [idx, text, embedding]

                # Если работаешь на GPU, можешь освободить память (опционально)
                torch.cuda.empty_cache()

            else:
                print(f'НЕТ ЦЕНЗУРЫ в строке {i}')

                text = 'Решение не прошло цензуру'
                # Получение эмбеддинга и его конвертация в строку
                embedding = embedding2string(get_sentence_embedding(text))

                # Заполнение DataFrame
                submit_df.loc[i] = [idx, text, embedding]

                # Если работаешь на GPU, можешь освободить память (опционально)
                torch.cuda.empty_cache()



        except Exception as e:
            print(f"Ошибка при обработке строки {i}: {e}")
            idx = test_solutions.iloc[i]['id']
            submit_df.loc[i] = [idx, "Ошибка", ""]


    # Сохранение в CSV
    submit_df.to_csv(save_path, index=False)
    print(f"Файл успешно сохранен по пути: {save_path}")
