import jax.numpy as jnp
import tensorflow as tf
import json
import os
from typing import Dict, Any, Tuple, List, Union

class DataLoader:
    def __init__(self, data_path: str, use_deepmind_format: bool = False, use_openai_format: bool = False,
                 max_steps: int = 8, use_calculation_annotations: bool = False):
        self.data_path = data_path
        self.use_deepmind_format = use_deepmind_format
        self.use_openai_format = use_openai_format
        self.max_steps = max_steps
        self.use_calculation_annotations = use_calculation_annotations

    def load_data(self) -> List[Dict[str, Any]]:
        if self.use_deepmind_format:
            return self._load_deepmind_data()
        elif self.use_openai_format:
            return self._load_openai_data()
        else:
            raise ValueError("Either use_deepmind_format or use_openai_format must be True")

    def _load_deepmind_data(self) -> List[Dict[str, Any]]:
        data = []
        for difficulty in ['train-easy', 'train-medium', 'train-hard']:
            file_path = os.path.join(self.data_path, f'{difficulty}.txt')
            with open(file_path, 'r') as f:
                for line in f:
                    question, answer = line.strip().split('\t')
                    data.append({
                        'question': question[:160],  # Limit to 160 characters
                        'answer': answer[:30],    # Limit to 30 characters
                        'difficulty': difficulty
                    })
        return data

    def _load_openai_data(self) -> List[Dict[str, Any]]:
        data = []
        file_path = os.path.join(self.data_path, 'train.jsonl')
        with open(file_path, 'r') as f:
            for line in f:
                problem = json.loads(line)
                steps = problem['answer'].split('\n')
                final_answer = steps[-1].split('####')[-1].strip()
                data.append({
                    'question': problem['question'],
                    'steps': steps[:-1],
                    'answer': final_answer,
                    'calculation_annotations': self._extract_calculations(problem['answer']) if self.use_calculation_annotations else None
                })
        return data

    def _extract_calculations(self, answer: str) -> List[str]:
        calculations = []
        for step in answer.split('\n'):
            if '<<' in step and '>>' in step:
                calc = step[step.index('<<')+2:step.index('>>')]
                calculations.append(calc)
        return calculations

    def preprocess_data(self, data: List[Dict[str, Any]]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        questions = [item['question'] for item in data]
        answers = [item['answer'] for item in data]

        # Convert to numerical representation (e.g., using tokenization)
        # This is a placeholder and should be replaced with actual tokenization logic
        question_tokens = jnp.array([[ord(c) for c in q] for q in questions])
        answer_tokens = jnp.array([[ord(c) for c in a] for a in answers])

        return question_tokens, answer_tokens

    def create_tf_dataset(self, data: List[Dict[str, Any]]) -> tf.data.Dataset:
        questions, answers = self.preprocess_data(data)
        dataset = tf.data.Dataset.from_tensor_slices((questions, answers))
        return dataset.batch(32).prefetch(tf.data.AUTOTUNE)

def create_data_loader(data_path: str, use_deepmind_format: bool = False, use_openai_format: bool = False,
                       max_steps: int = 8, use_calculation_annotations: bool = False) -> DataLoader:
    return DataLoader(data_path, use_deepmind_format, use_openai_format, max_steps, use_calculation_annotations)
