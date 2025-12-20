from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from fishfarm.models import GenerationRequest, Message, Model
from fishfarm.tasks.base import Task, TaskResult


def get_accuracy(generated_text: str, target_text: str) -> int:
    if str(generated_text).strip().strip(":`'\"(.) ").lower() == str(target_text).strip().strip(":`'\"(.) ").lower():
        return 1
    try:
        if str(float(generated_text)) == str(target_text).strip():  # For example, "1.0" == "1"
            return 1
        return 0
    except:
        return 0


def get_choice(txt: str) -> str:
    txt = str(txt).strip().strip(":`'\"(.) ").lower()
    CHOICES = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    ]
    for choice in CHOICES:
        if txt.startswith(choice):
            return choice


def get_choice_accuracy(generated_text: str, target_text: str) -> int:
    if get_choice(generated_text) == get_choice(target_text):
        return 1
    return 0


def get_bool_value_from_text(text):
    """Returns None if there was no meaningful boolean value that could be found."""
    # Convert to string if necessary.
    text = str(text)
    if "1" in text:
        return True
    if "0" in text:
        return False
    if "yes" in text.lower():
        return True
    if "no" in text.lower():
        return False
    if "true" in text.lower():
        return True
    if "false" in text.lower():
        return False
    if "positive" in text.lower():
        return True
    if "negative" in text.lower():
        return False
    if "valid" in text.lower():
        return True
    if "invalid" in text.lower():
        return False
    return None


def get_binary_accuracy_flex(generated_text, target_text):
    """Returns 1 if the generated text and target text are equal in boolean space.

    This is a flexible matching function that can handle a variety of boolean representations.
    - yes/no
    - true/false
    - 1/0
    - positive/negative
    - valid/invalid
    """
    generated_prediction = get_bool_value_from_text(generated_text)
    if generated_prediction is None:
        return 0

    target_prediction = get_bool_value_from_text(target_text)
    if target_prediction is None:
        return 0

    return int(generated_prediction == target_prediction)


@dataclass
class QASample:
    question: str
    answer: str


class QATask(Task):
    def __init__(
        self,
        samples: Sequence[QASample],
        eval_fn: Callable,
        context_messages: Sequence[Message] = (),
    ) -> None:
        self.samples = list(samples)
        self.eval_fn = eval_fn
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def batch_evaluate_with_outputs(self, outputs: Sequence[str], sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        """Evaluate using pre-generated outputs for batch evaluation."""
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        
        assert len(outputs) == len(samples), f"Number of outputs ({len(outputs)}) must match number of samples ({len(samples)})"

        sample_details = []
        for sample, output in zip(samples, outputs):
            is_correct = self.eval_fn(output, sample.answer)
            details = dict(
                problem=sample.question,
                output=output,
                answer=sample.answer,
                is_correct=is_correct,
                # Expose the user input/instruction explicitly for downstream analysis
                user_input=sample.question,
                user_instruction=sample.question,
            )
            sample_details.append(details)

        agg_metrics = dict(acc=sum(sample["is_correct"] for sample in sample_details) / len(sample_details))
        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.question))
            requests.append(GenerationRequest(messages=messages))

        # Generate outputs
        outputs = [result.generation for result in model.generate(requests)]
        
        # Use batch evaluation method
        return self.batch_evaluate_with_outputs(outputs, sample_ids)


@dataclass
class RatingPredictionSample:
    """Sample for rating prediction tasks (e.g., ratings from 1-5).

    Optional metadata mirrors fields available in LaMP/LongLaMP datasets so we
    can surface user context alongside evaluation outputs.
    """
    prompt: str
    rating: float  # Expected rating as float
    user_profile: str | None = None
    user_id: str | None = None
    question_id: str | None = None


@dataclass
class ClassificationSample:
    """Sample for classification tasks.

    Extend with optional user metadata for downstream analysis.
    """
    prompt: str
    label: str  # Expected classification label
    user_profile: str | None = None
    user_id: str | None = None
    question_id: str | None = None


class LaMPRatingTask(Task):
    """LaMP evaluation task for rating prediction with MAE and RMSE metrics."""
    
    def __init__(
        self,
        samples: Sequence[RatingPredictionSample],
        context_messages: Sequence[Message] = (),
    ) -> None:
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def batch_evaluate_with_outputs(self, outputs: Sequence[str], sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        """Evaluate using pre-generated outputs for batch evaluation."""
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        
        assert len(outputs) == len(samples), f"Number of outputs ({len(outputs)}) must match number of samples ({len(samples)})"
        
        sample_details = []
        predicted_ratings = []
        target_ratings = []
        stsb_predicted_ratings = []
        stsb_target_ratings = []
        
        from .metric_fns import get_numeric_value, get_stsb_number
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        import numpy as np
        
        for sample, output in zip(samples, outputs):
            # Extract numeric values for MAE/RMSE calculation
            predicted_value = get_numeric_value(output)
            target_value = float(sample.rating)
            
            # Extract numeric values for STSB calculation (using get_stsb_number for consistency)
            stsb_predicted_value = get_stsb_number(output)
            stsb_target_value = get_stsb_number(str(sample.rating))
            
            # Only include samples where we can extract a valid prediction
            if predicted_value is not None:
                predicted_ratings.append(predicted_value)
                target_ratings.append(target_value)
            
            # Only include samples where we can extract valid STSB values
            if stsb_predicted_value is not None and stsb_target_value is not None:
                stsb_predicted_ratings.append(stsb_predicted_value)
                stsb_target_ratings.append(stsb_target_value)
            
            details = dict(
                problem=sample.prompt,
                output=output,
                target_rating=sample.rating,
                predicted_value=predicted_value,
                stsb_predicted_value=stsb_predicted_value,
                # Include explicit input/instruction string
                user_input=sample.prompt,
                user_instruction=sample.prompt,
            )
            if getattr(sample, "user_profile", None) is not None:
                details["user_profile"] = sample.user_profile
            if getattr(sample, "user_id", None) is not None:
                details["user_id"] = sample.user_id
            if getattr(sample, "question_id", None) is not None:
                details["question_id"] = sample.question_id
            sample_details.append(details)

        # Calculate aggregate metrics using sklearn in batch
        if predicted_ratings and target_ratings:
            mae = mean_absolute_error(target_ratings, predicted_ratings)
            mse = mean_squared_error(target_ratings, predicted_ratings)
            rmse = np.sqrt(mse)
        else:
            mae = 0.0
            rmse = 0.0
        
        # Calculate STSB score using entire dataset
        if stsb_predicted_ratings and stsb_target_ratings:
            # Calculate normalized MAE for STSB (following the original get_stsb logic)
            stsb_mae = mean_absolute_error(stsb_target_ratings, stsb_predicted_ratings)
            normalized_stsb_mae = stsb_mae / 5.0  # Normalize by max STSB score (5)
            stsb = 1.0 - normalized_stsb_mae  # Convert to accuracy-like score
        else:
            stsb = 0.0
        
        agg_metrics = dict(mae=mae, rmse=rmse, stsb=stsb)
        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.prompt))
            requests.append(GenerationRequest(messages=messages))

        # Generate outputs
        outputs = [result.generation for result in model.generate(requests)]
        
        # Use batch evaluation method
        return self.batch_evaluate_with_outputs(outputs, sample_ids)


class LaMPClassificationTask(Task):
    """LaMP evaluation task for classification with accuracy and F1 metrics."""
    
    def __init__(
        self,
        samples: Sequence[ClassificationSample],
        context_messages: Sequence[Message] = (),
    ) -> None:
        self.samples = list(samples)
        self.context_messages = context_messages

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def batch_evaluate_with_outputs(self, outputs: Sequence[str], sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        """Evaluate using pre-generated outputs for batch evaluation."""
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        
        assert len(outputs) == len(samples), f"Number of outputs ({len(outputs)}) must match number of samples ({len(samples)})"
        
        sample_details = []
        predicted_labels = []
        target_labels = []
        
        from .metric_fns import get_classification_label
        from sklearn.metrics import accuracy_score, f1_score
        
        for sample, output in zip(samples, outputs):
            # Normalize both predicted and target labels for comparison
            predicted_label = output.strip().lower()
            target_label = sample.label.strip().lower()
            
            # Include all samples with valid predictions
            predicted_labels.append(predicted_label)
            target_labels.append(target_label)
            is_correct = 1 if predicted_label == target_label else 0
            
            details = dict(
                problem=sample.prompt,
                output=output,
                target_label=sample.label,
                predicted_label=predicted_label,
                is_correct=is_correct,
                # Include explicit input/instruction string
                user_input=sample.prompt,
                user_instruction=sample.prompt,
            )
            if getattr(sample, "user_profile", None) is not None:
                details["user_profile"] = sample.user_profile
            if getattr(sample, "user_id", None) is not None:
                details["user_id"] = sample.user_id
            if getattr(sample, "question_id", None) is not None:
                details["question_id"] = sample.question_id
            sample_details.append(details)

        # Calculate aggregate metrics using sklearn in batch
        if predicted_labels and target_labels:
            accuracy = accuracy_score(target_labels, predicted_labels)
            # Use 'weighted' average for F1 to handle multi-class scenarios
            f1 = f1_score(target_labels, predicted_labels, average='weighted', zero_division=0)
        else:
            accuracy = 0.0
            f1 = 0.0
        
        agg_metrics = dict(acc=accuracy, f1=f1)
        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.prompt))
            requests.append(GenerationRequest(messages=messages))

        # Generate outputs
        outputs = [result.generation for result in model.generate(requests)]
        
        # Use batch evaluation method
        return self.batch_evaluate_with_outputs(outputs, sample_ids)


@dataclass
class TextGenerationSample:
    """Sample for text generation tasks with reference text.

    Extended to optionally carry user metadata when available, so eval JSON can
    include fields like user_profile, user_id, and question_id.
    """
    prompt: str
    reference: str  # Expected/reference text for evaluation
    # Optional user metadata (present for personalized datasets like PersonalReddit/EC)
    user_profile: str | None = None
    user_id: str | None = None
    question_id: str | None = None


from tqdm import tqdm

class LaMPTextGenerationTask(Task):
    """LaMP evaluation task for text generation with ROUGE and METEOR metrics."""
    
    def __init__(
        self,
        samples: Sequence[TextGenerationSample],
        context_messages: Sequence[Message] = (),
        rouge_types: tuple = ("rouge1", "rougeL"),
    ) -> None:
        self.samples = list(samples)
        self.context_messages = context_messages
        self.rouge_types = rouge_types

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def batch_evaluate_with_outputs(self, outputs: Sequence[str], sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        """Evaluate using pre-generated outputs for batch evaluation."""
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        
        assert len(outputs) == len(samples), f"Number of outputs ({len(outputs)}) must match number of samples ({len(samples)})"
        
        sample_details = []
        rouge_scores = {f"{rt}_{metric}": [] for rt in self.rouge_types for metric in ["precision", "recall", "fmeasure"]}
        meteor_scores = []
        
        # Import metrics
        import torch
        from .metric_fns import get_meteor
        from torchmetrics.text.rouge import ROUGEScore
        
        # Initialize ROUGE scorer with GPU acceleration if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device for ROUGE:", device)
        rouge_scorer = ROUGEScore(rouge_keys=self.rouge_types).to(device)
        
        # Compute ROUGE in batch
        references = [sample.reference for sample in samples]
        rouge_result = rouge_scorer(outputs, references)
        
        # Compute METEOR scores (not batched, as get_meteor may not support batch)
        print("Computing METEOR scores...")
        for i in tqdm(range(len(samples)), desc="Scoring samples"):
            output = outputs[i]
            sample = samples[i]
            meteor_score = get_meteor(output, sample.reference)
            meteor_scores.append(meteor_score)
            
            details = dict(
                problem=sample.prompt,
                output=output,
                reference=sample.reference,
                meteor=meteor_score,
                # Include explicit input/instruction string
                user_input=sample.prompt,
                user_instruction=sample.prompt,
            )

            # If available, include user metadata for downstream analysis
            if getattr(sample, "user_profile", None) is not None:
                details["user_profile"] = sample.user_profile
            if getattr(sample, "user_id", None) is not None:
                details["user_id"] = sample.user_id
            if getattr(sample, "question_id", None) is not None:
                details["question_id"] = sample.question_id
            
            # Add ROUGE scores to details and collect for aggregation
            for rouge_type in self.rouge_types:
                for metric in ["precision", "recall", "fmeasure"]:
                    key = f"{rouge_type}_{metric}"
                    # Handle both batched and scalar rouge results
                    rouge_tensor = rouge_result[key]
                    if rouge_tensor.dim() == 0:  # 0-dimensional tensor (scalar)
                        score = rouge_tensor.item()
                    else:  # Tensor with batch dimension
                        score = rouge_tensor[i].item() if hasattr(rouge_tensor[i], 'item') else float(rouge_tensor[i])
                    details[key] = score
                    rouge_scores[key].append(score)
            
            sample_details.append(details)

        # Calculate aggregate metrics
        agg_metrics = dict()
        
        # ROUGE aggregates
        for key, scores in rouge_scores.items():
            agg_metrics[key] = sum(scores) / len(scores) if scores else 0.0
        
        # METEOR aggregate
        agg_metrics["meteor"] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        
        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]
        requests = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.append(Message(role="user", content=sample.prompt))
            requests.append(GenerationRequest(messages=messages))

        # Generate outputs
        outputs = []
        for result in tqdm(model.generate(requests), total=len(samples), desc="Generating outputs"):
            outputs.append(result.generation)
        
        # Use batch evaluation method
        return self.batch_evaluate_with_outputs(outputs, sample_ids)


TASK_EVAL_FNS = {
    "winogrande": get_choice_accuracy,
    "boolq": get_binary_accuracy_flex,
    "piqa": get_choice_accuracy,
    "hellaswag": get_choice_accuracy,
    "arc": get_choice_accuracy,
    "openbookqa": get_choice_accuracy,
}


# Conversation-based text generation for chat-style datasets (e.g., PRISM)
@dataclass
class ChatConversationSample:
    """Sample containing a full conversation history and a reference response.

    Extended to carry additional metadata for richer eval result logging.
    """
    messages: list[Message]
    reference: str
    # Additional optional metadata
    input_messages: list[dict] | None = None  # raw list of {role, content} used before chat template encoding
    user_input: str | None = None  # last user turn content (for convenience)
    user_profile: str | None = None
    user_id: str | None = None
    question_id: str | None = None
    # Additional fields for comprehensive logging
    original_sample_data: dict | None = None  # Store all original dataset fields


class ConversationTextGenerationTask(Task):
    """Text generation task where the prompt is a conversation, not a single string."""

    def __init__(
        self,
        samples: Sequence[ChatConversationSample],
        context_messages: Sequence[Message] = (),
        rouge_types: tuple = ("rouge1", "rougeL"),
    ) -> None:
        self.samples = list(samples)
        self.context_messages = context_messages
        self.rouge_types = rouge_types

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def batch_evaluate_with_outputs(self, outputs: Sequence[str], sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        assert len(outputs) == len(samples), f"Number of outputs ({len(outputs)}) must match number of samples ({len(samples)})"

        sample_details = []
        rouge_scores = {f"{rt}_{metric}": [] for rt in self.rouge_types for metric in ["precision", "recall", "fmeasure"]}
        meteor_scores = []

        # Import metrics
        import torch
        from .metric_fns import get_meteor
        from torchmetrics.text.rouge import ROUGEScore

        # Initialize ROUGE scorer with GPU acceleration if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device for ROUGE:", device)
        rouge_scorer = ROUGEScore(rouge_keys=self.rouge_types).to(device)

        # Compute ROUGE in batch
        references = [sample.reference for sample in samples]
        rouge_result = rouge_scorer(outputs, references)

        # Compute METEOR scores
        from tqdm import tqdm
        print("Computing METEOR scores...")
        for i in tqdm(range(len(samples)), desc="Scoring samples"):
            output = outputs[i]
            sample = samples[i]
            meteor_score = get_meteor(output, sample.reference)
            meteor_scores.append(meteor_score)

            # Build raw input messages and determine last user input
            raw_input_messages = sample.input_messages
            if raw_input_messages is None:
                raw_input_messages = [{"role": m.role, "content": m.content} for m in sample.messages]

            last_user_input = sample.user_input
            if last_user_input is None:
                for m in reversed(raw_input_messages):
                    if m.get("role") == "user":
                        last_user_input = m.get("content", "")
                        break

            details = dict(
                output=output,
                reference=sample.reference,
                meteor=meteor_score,
                user_input=raw_input_messages,  # list of messages prior to chat template encoding
                last_user_input=last_user_input,
                # Also include a compact alias for last_user_input
                user_instruction=last_user_input,
                user_profile=sample.user_profile,
                user_id=sample.user_id,
                question_id=sample.question_id,
            )
            
            # Include all original sample data if available (for ALOE/PRISM comprehensive logging)
            if hasattr(sample, 'original_sample_data'):
                details['original_sample_data'] = sample.original_sample_data
                
                # Also add profile-related fields at top level for easy access
                sample_data = sample.original_sample_data
                if 'profile_all_history' in sample_data:
                    details['profile_all_history'] = sample_data['profile_all_history']
                
                # Add any profile retrieval fields
                for key, value in sample_data.items():
                    if key.startswith('profile_retrieval_k'):
                        details[key] = value

            # Add ROUGE scores to details and collect for aggregation
            for rouge_type in self.rouge_types:
                for metric in ["precision", "recall", "fmeasure"]:
                    key = f"{rouge_type}_{metric}"
                    rouge_tensor = rouge_result[key]
                    if rouge_tensor.dim() == 0:
                        score = rouge_tensor.item()
                    else:
                        score = rouge_tensor[i].item() if hasattr(rouge_tensor[i], 'item') else float(rouge_tensor[i])
                    details[key] = score
                    rouge_scores[key].append(score)

            sample_details.append(details)

        # Calculate aggregate metrics
        agg_metrics = dict()
        for key, scores in rouge_scores.items():
            agg_metrics[key] = sum(scores) / len(scores) if scores else 0.0
        agg_metrics["meteor"] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

        return TaskResult(aggregate_metrics=agg_metrics, sample_details=sample_details)

    def evaluate(self, model: Model, sample_ids: Optional[Sequence[int]] = None) -> TaskResult:
        if sample_ids is None:
            sample_ids = range(len(self.samples))
        samples = [self.samples[sample_id] for sample_id in sample_ids]

        # Build requests by concatenating context + conversation
        requests: list[GenerationRequest] = []
        for sample in samples:
            messages = list(self.context_messages)
            messages.extend(sample.messages)
            requests.append(GenerationRequest(messages=messages))

        # Generate outputs
        from tqdm import tqdm
        outputs = []
        for result in tqdm(model.generate(requests), total=len(samples), desc="Generating outputs"):
            outputs.append(result.generation)

        # Batch score
        return self.batch_evaluate_with_outputs(outputs, sample_ids)
