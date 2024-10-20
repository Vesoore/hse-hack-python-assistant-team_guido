import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, PreTrainedModel
from typing import List, Dict, Any
import numpy as np



class ToxiClassifier:
    def __init__(self, model_name: str = 'Skoltech/russian-inappropriate-messages') -> None:
        # Load tokenizer and model
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(model_name)
        # Define labels
        self.labels: List[str] = ['OK', 'Toxic', 'Severe toxic', 'Risks']
        # Get max number of tokens that the model can handle
        self.max_length: int = self.model.config.max_position_embeddings

    def predict(self, text: str) -> Dict[str, Any]:
        # Tokenize text and get list of tokens
        inputs: Dict[str, torch.Tensor] = self.tokenizer(text, return_tensors='pt', truncation=False)
        input_ids: torch.Tensor = inputs['input_ids'][0]

        # Split tokens into chunks of max length
        chunks: List[torch.Tensor] = input_ids.split(self.max_length - 2)  # -2 for [CLS] and [SEP] tokens

        # Flag to determine toxicity
        is_toxic: bool = False
        all_probabilities: List[np.ndarray] = []

        # Iterate over all chunks
        for chunk in chunks:
            # Add special tokens [CLS] and [SEP]
            chunk = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                chunk,
                torch.tensor([self.tokenizer.sep_token_id])
            ])

            # Convert back to batch with dimension [1, sequence_length]
            chunk_input: Dict[str, torch.Tensor] = {'input_ids': chunk.unsqueeze(0)}

            # Inference without gradients
            with torch.inference_mode():
                logits: torch.Tensor = self.model(**chunk_input).logits

            # Convert logits to probabilities
            probas: np.ndarray = torch.softmax(logits, dim=-1)[0].cpu().detach().numpy()
            all_probabilities.append(probas)

            # Determine class with highest probability
            predicted_class_idx: int = int(torch.argmax(torch.tensor(probas)).item())
            predicted_label: str = self.labels[predicted_class_idx]

            # If at least one chunk is toxic or severe toxic, mark as toxic
            if predicted_label in ['Toxic', 'Severe toxic', 'Risks']:
                is_toxic = True

        # Determine final label
        final_label: str = 'Toxic' if is_toxic else 'OK'

        # Return class label and probabilities for each chunk
        return {
            'label': final_label,
            'probabilities_per_chunk': all_probabilities
        }


if __name__ == "__main__":

    text = 'Я тебя люблю!'

    classifier = ToxiClassifier()

    result = classifier.predict(text)

    print(f"Предсказанный класс: {result['label']}")
