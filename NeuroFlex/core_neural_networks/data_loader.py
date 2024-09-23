import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
import logging
import time
from typing import Dict, Any, Tuple, List, Union


class DataLoader(nn.Module):
    def __init__(
        self,
        data_path: str,
        use_deepmind_format: bool = False,
        use_openai_format: bool = False,
        max_steps: int = 8,
        use_calculation_annotations: bool = False,
        learning_rate: float = 0.001,
    ):
        super(DataLoader, self).__init__()
        self.data_path = data_path
        self.use_deepmind_format = use_deepmind_format
        self.use_openai_format = use_openai_format
        self.max_steps = max_steps
        self.use_calculation_annotations = use_calculation_annotations
        self.learning_rate = learning_rate
        self.performance_threshold = 0.8
        self.update_interval = 86400  # 24 hours in seconds
        self.gradient_norm_threshold = 10
        self.performance_history_size = 100
        self.is_trained = False
        self.performance = 0.0
        self.last_update = 0
        self.gradient_norm = 0
        self.performance_history = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Placeholder forward method
        return x

    def load_data(self) -> List[Dict[str, Any]]:
        if self.use_deepmind_format:
            return self._load_deepmind_data()
        elif self.use_openai_format:
            return self._load_openai_data()
        else:
            raise ValueError(
                "Either use_deepmind_format or use_openai_format must be True"
            )

    def _load_deepmind_data(self) -> List[Dict[str, Any]]:
        data = []
        for difficulty in ["train-easy", "train-medium", "train-hard"]:
            file_path = os.path.join(self.data_path, f"{difficulty}.txt")
            with open(file_path, "r") as f:
                for line in f:
                    question, answer = line.strip().split("\t")
                    data.append(
                        {
                            "question": question[:160],  # Limit to 160 characters
                            "answer": answer[:30],  # Limit to 30 characters
                            "difficulty": difficulty,
                        }
                    )
        return data

    def _load_openai_data(self) -> List[Dict[str, Any]]:
        data = []
        file_path = os.path.join(self.data_path, "train.jsonl")
        with open(file_path, "r") as f:
            for line in f:
                problem = json.loads(line)
                steps = problem["answer"].split("\n")
                final_answer = steps[-1].split("####")[-1].strip()
                data.append(
                    {
                        "question": problem["question"],
                        "steps": steps[:-1],
                        "answer": final_answer,
                        "calculation_annotations": (
                            self._extract_calculations(problem["answer"])
                            if self.use_calculation_annotations
                            else None
                        ),
                    }
                )
        return data

    def _extract_calculations(self, answer: str) -> List[str]:
        calculations = []
        for step in answer.split("\n"):
            if "<<" in step and ">>" in step:
                calc = step[step.index("<<") + 2 : step.index(">>")]
                calculations.append(calc)
        return calculations

    def preprocess_data(
        self, data: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        questions = [item["question"] for item in data]
        answers = [item["answer"] for item in data]

        # Convert to numerical representation (e.g., using tokenization)
        # This is a placeholder and should be replaced with actual tokenization logic
        question_tokens = torch.tensor(
            [[ord(c) for c in q] for q in questions], dtype=torch.long
        )
        answer_tokens = torch.tensor(
            [[ord(c) for c in a] for a in answers], dtype=torch.long
        )

        return question_tokens.to(self.device), answer_tokens.to(self.device)

    def create_dataloader(
        self, data: List[Dict[str, Any]], batch_size: int = 32
    ) -> torch.utils.data.DataLoader:
        questions, answers = self.preprocess_data(data)
        dataset = torch.utils.data.TensorDataset(questions, answers)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4
        )

    def diagnose(self):
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        if self.gradient_norm > self.gradient_norm_threshold:
            issues.append("Gradient explosion detected")
        if len(self.performance_history) > 5 and all(
            p < 0.01 for p in self.performance_history[-5:]
        ):
            issues.append("Model is stuck in local minimum")
        return issues

    def heal(self, issues):
        for issue in issues:
            if issue == "Model is not trained":
                logging.info("Model needs training")
            elif issue == "Model performance is below threshold":
                self.improve_model()
            elif issue == "Model hasn't been updated in 24 hours":
                self.update_model()
            elif issue == "Gradient explosion detected":
                self.handle_gradient_explosion()
            elif issue == "Model is stuck in local minimum":
                self.escape_local_minimum()

    def improve_model(self):
        logging.info("Improving model performance...")
        self.performance = min(self.performance * 1.2 + 0.01, 1.0)
        self.update_performance()

    def update_model(self):
        logging.info("Updating model...")
        self.last_update = time.time()
        self.update_performance()

    def handle_gradient_explosion(self):
        logging.info("Handling gradient explosion...")
        self.learning_rate *= 0.5

    def escape_local_minimum(self):
        logging.info("Attempting to escape local minimum...")
        self.learning_rate = min(self.learning_rate * 2.5, 0.1)
        logging.info(f"New learning rate: {self.learning_rate}")

    def update_performance(self):
        self.performance_history.append(self.performance)
        if len(self.performance_history) > self.performance_history_size:
            self.performance_history.pop(0)

    def adjust_learning_rate(self):
        if len(self.performance_history) >= 2:
            current_performance = self.performance_history[-1]
            previous_performance = self.performance_history[-2]

            if current_performance > previous_performance:
                self.learning_rate *= 1.05
            elif current_performance < previous_performance:
                self.learning_rate *= 0.95
        else:
            self.learning_rate *= 1.01

        self.learning_rate = max(min(self.learning_rate, 0.1), 1e-5)
        return self.learning_rate


def create_data_loader(
    data_path: str,
    use_deepmind_format: bool = False,
    use_openai_format: bool = False,
    max_steps: int = 8,
    use_calculation_annotations: bool = False,
) -> DataLoader:
    return DataLoader(
        data_path,
        use_deepmind_format,
        use_openai_format,
        max_steps,
        use_calculation_annotations,
    )
