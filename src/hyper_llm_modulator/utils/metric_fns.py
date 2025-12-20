import json
import re
import math
import ssl
import torch

# Disable SSL verification for NLTK downloads (workaround for certificate issues)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import inflect
from torchmetrics.text.rouge import ROUGEScore

# Global ROUGE scorer instance for efficiency and GPU acceleration
_rouge_scorer = None

def get_rouge_scorer():
    """Get or create a shared ROUGE scorer instance with GPU acceleration if available."""
    global _rouge_scorer
    if _rouge_scorer is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _rouge_scorer = ROUGEScore().to(device)
    return _rouge_scorer


def get_accuracy(generated_text, target_text):
    if str(generated_text).strip(".").lower() == str(target_text).strip(".").lower():
        return 1
    try:
        if str(float(generated_text)) == str(target_text).strip():  # For example, "1.0" == "1"
            return 1
        return 0
    except:
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
    return None


def get_binary_accuracy_flex(generated_text, target_text):
    """Returns 1 if the generated text and target text are equal in boolean space.

    This is a flexible matching function that can handle a variety of boolean representations.
    - yes/no
    - true/false
    - 1/0
    - positive/negative
    """
    generated_prediction = get_bool_value_from_text(generated_text)
    if generated_prediction is None:
        # print("Could not extract a boolean prediction from the generated text.")
        return 0

    target_prediction = get_bool_value_from_text(target_text)
    if target_prediction is None:
        # print("Could not extract a boolean prediction from the target text.")
        return 0

    return int(generated_prediction == target_prediction)


def get_mrpc_accuracy(generated_text, target_text):
    """Tries to match based on vanilla accuracy. If not, then check for the consistent presence of (1 | 2)."""
    vanilla_accuracy = get_accuracy(generated_text, target_text)
    if vanilla_accuracy == 1:
        return 1

    # If the vanilla accuracy is 0, then we need to check other heuristics.
    if "1" in generated_text and "1" in str(target_text):
        return 1

    if "2" in generated_text and "2" in str(target_text):
        return 1

    return 0


def get_mnli_accuracy(generated_text, target_text):
    """Tries to match based on vanilla accuracy. If not, then check for the consistent presence of (0 | 1 | 2)."""
    vanilla_accuracy = get_accuracy(generated_text, target_text)
    if vanilla_accuracy == 1:
        return 1

    # If the vanilla accuracy is 0, then we need to check other heuristics.
    if "0" in generated_text and "0" in str(target_text):
        return 1

    if "1" in generated_text and "1" in str(target_text):
        return 1

    if "2" in generated_text and "2" in str(target_text):
        return 1

    return 0


def get_hellaswag_accuracy(generated_text, target_text):
    """Tries to match based on vanilla accuracy. If not, then check for the consistent presence of (0 | 1 | 2 | 3)."""
    vanilla_accuracy = get_accuracy(generated_text, target_text)
    if vanilla_accuracy == 1:
        return 1

    # If the vanilla accuracy is 0, then we need to check other heuristics.
    if "0" in generated_text and "0" in str(target_text):
        return 1

    if "1" in generated_text and "1" in str(target_text):
        return 1

    if "2" in generated_text and "2" in str(target_text):
        return 1

    if "3" in generated_text and "3" in str(target_text):
        return 1

    return 0


def get_rouge(generated_text, target_text):
    """Compute ROUGE-L F-measure using GPU-accelerated scorer."""
    rouge_scorer = get_rouge_scorer()
    
    # Convert texts to list format as expected by the scorer
    preds = [str(generated_text)]
    targets = [str(target_text)]
    
    # Compute ROUGE score
    result = rouge_scorer(preds, targets)
    return result["rougeL_fmeasure"].item() if hasattr(result["rougeL_fmeasure"], 'item') else result["rougeL_fmeasure"]

def get_rouge_batch(generated_texts, target_texts):
    """Compute ROUGE scores for multiple text pairs in a batch for better GPU utilization."""
    rouge_scorer = get_rouge_scorer()
    
    # Convert to string lists
    preds = [str(text) for text in generated_texts]
    targets = [str(text) for text in target_texts]
    
    # Compute ROUGE scores in batch
    result = rouge_scorer(preds, targets)
    scores = result["rougeL_fmeasure"]
    
    # Convert to list of individual scores
    if hasattr(scores, 'tolist'):
        return scores.tolist()
    elif hasattr(scores, 'item'):
        return [scores.item()]
    else:
        return [scores] if not isinstance(scores, list) else scores


def get_first_number(text):
    # Pattern explanation:
    # \d{1,3}(?:,\d{3})* - Matches numbers with commas for thousands
    # (?:\.\d+)? - Optional decimal part
    pattern = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"

    # Find all matches in the string
    matches = re.findall(pattern, text)

    # Return the first match or None if no match is found
    return matches[0] if matches else None


def get_stsb_number(text):
    """Returns a number from 0 to 5.

    If the text is not just the number, then we try to find the first number in the text.

    If there is no number in the text, then we return None.
    If the number is greater than 5, then we return None.
    If the number is less than 0, then we return None.
    """
    try:
        return float(text)
    except:
        first_number = get_first_number(str(text))
        if first_number is None:
            return None

        number = float(first_number)
        if number > 5:
            return None
        if number < 0:
            return None

        return float(first_number)


def get_stsb(generated_text, target_text):
    # Return 1 - percentage error.
    generated_text_number = get_stsb_number(generated_text)
    target_text_number = get_stsb_number(target_text)

    if generated_text_number is None:
        return 0

    # Normalize by dividing by 5, which is maximum.
    mean_absolute_error = abs(generated_text_number - target_text_number) / 5
    return 1 - mean_absolute_error


def get_dbpedia(generated_text, target_text):
    classes = [
        "Company",
        "EducationalInstitution",
        "Artist",
        "Athlete",
        "OfficeHolder",
        "MeanOfTransportation",
        "Building",
        "NaturalPlace",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "WrittenWork",
    ]
    try:
        generated_text = int(generated_text)
        generated_text = classes[generated_text]
    finally:
        return get_rouge(generated_text, target_text)


def get_drop(generated_text, target_text):
    engine = inflect.engine()
    start = target_text.index("[") + 2
    end = target_text.index("]") - 1
    target_text = target_text[start:end].lower()
    generated_text = generated_text.lower()

    # Example: "{'spans': array(['86598'], dtype=object),
    #            'types': array(['number'], dtype=object)}" --> "86598"
    if generated_text.startswith("{'spans': array(['"):
        start = generated_text.index("[") + 2
        end = generated_text.index("]") - 1
        generated_text = generated_text[start:end]

    # Example: "1" vs. "one"
    try:
        target_text = int(target_text)
        target_text = engine.number_to_words(target_text)
    except:
        pass
    try:
        generated_text = int(generated_text)
        generated_text = engine.number_to_words(generated_text)
    finally:
        return get_rouge(generated_text, target_text)


def get_label_and_explanation(generated_text, target_text):
    """For use with datasets created with join_explanations.py."""
    try:
        generated_json = json.loads(generated_text)
        target_json = json.loads(target_text)

        # Check assertions.
        assert len(generated_json) == len(target_json) == 2
        assert "explanation" in generated_json
        assert "explanation" in target_json

        generated_label = list(generated_json.keys() - {"explanation"})[0]
        target_label = list(target_json.keys() - {"explanation"})[0]
    except Exception as e:
        # print(f"Encountered exception during label_and_explanation metric: {e}")
        return 0

    if generated_json[generated_label] == target_json[target_label]:
        # The labels are correct. Now check the explanation ROUGE score.
        return get_rouge(generated_json["explanation"], target_json["explanation"])

    # If the label is inaccurate, then the explanation is irrelevant.
    return 0


def find_last_number(s):
    # Pattern explanation:
    # [\$€£]? - Optional currency symbols ($, €, £)
    # [+-]? - Optional sign (plus or minus)
    # \d{1,3}(?:,\d{3})* - Matches numbers with commas for thousands
    # (?:\.\d+)? - Optional decimal part
    pattern = r"[\$€£]?[+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"

    # Find all matches in the string
    matches = re.findall(pattern, s)

    # Return the last match or None if no match is found
    return matches[-1] if matches else None


def get_gsm8k_regex(generated_text, target_text):
    pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"

    try:
        generated_match = "".join(re.findall(pattern, generated_text)[-1])
        target_match = "".join(re.findall(pattern, target_text)[-1])
    except IndexError:
        return 0

    if generated_match == None or target_match == None:
        return 0
    try:
        if float(generated_match.strip("$.").replace(",", "")) == float(target_match.strip("$.").replace(",", "")):
            return 1
    except ValueError:
        pass
    return 0


def get_customer_support_accuracy(generated_text, target_text):
    vanilla_accuracy = get_accuracy(generated_text, target_text)
    if vanilla_accuracy == 1:
        return 1
    if target_text.lower() in generated_text.lower():
        return 1
    return 0


def get_numeric_value(text):
    """Extract numeric value from text for rating prediction tasks.
    
    Returns None if no valid numeric value could be found.
    """
    text = str(text).strip()
    
    # Try direct float conversion first
    try:
        return float(text)
    except ValueError:
        pass
    
    # Look for first number in the text
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    if matches:
        try:
            return float(matches[0])
        except ValueError:
            pass
    
    return None


def get_mae(generated_text, target_text):
    """Calculate Mean Absolute Error between generated and target ratings.
    
    Returns 0 if either value cannot be parsed as a number.
    """
    generated_value = get_numeric_value(generated_text)
    target_value = get_numeric_value(target_text)
    
    if generated_value is None or target_value is None:
        return 0.0
    
    return abs(float(generated_value) - float(target_value))


def get_rmse(generated_text, target_text):
    """Calculate Root Mean Square Error between generated and target ratings.
    
    Returns 0 if either value cannot be parsed as a number.
    """
    generated_value = get_numeric_value(generated_text)
    target_value = get_numeric_value(target_text)
    
    if generated_value is None or target_value is None:
        return 0.0
    
    return math.sqrt((float(generated_value) - float(target_value)) ** 2)


def get_f1_score(generated_text, target_text, labels=None):
    """Calculate F1 score for classification tasks.
    
    For binary classification, calculates F1 directly.
    For multi-class, can accept a list of labels to consider.
    """
    # Clean and normalize text
    generated_text = str(generated_text).strip().lower()
    target_text = str(target_text).strip().lower()
    
    # For exact match
    if generated_text == target_text:
        return 1.0
    
    # Try to extract classification labels
    generated_label = get_classification_label(generated_text)
    target_label = get_classification_label(target_text)
    
    if generated_label is None or target_label is None:
        return 0.0
    
    # For binary classification or exact match
    if generated_label == target_label:
        return 1.0
    else:
        return 0.0


def get_classification_label(text):
    """Extract classification label from text output.
    
    Looks for common label patterns in model outputs.
    """
    text = str(text).strip().lower()
    
    # Look for quoted labels
    quoted_pattern = r"['\"]([^'\"]+)['\"]"
    quoted_matches = re.findall(quoted_pattern, text)
    if quoted_matches:
        return quoted_matches[0]
    
    # Look for labels after common prefixes
    prefixes = ["label:", "class:", "category:", "answer:", "prediction:"]
    for prefix in prefixes:
        if prefix in text:
            after_prefix = text.split(prefix, 1)[1].strip()
            # Get first word after prefix
            words = after_prefix.split()
            if words:
                return words[0].strip(".,!?()")
    
    # Look for single word at the beginning
    words = text.split()
    if words:
        first_word = words[0].strip(".,!?()")
        if len(first_word) > 0:
            return first_word
    
    return None


def get_meteor(generated_text, target_text):
    """Calculate METEOR score between generated and target text.
    
    METEOR is a metric for evaluating machine translation and text generation
    that considers precision, recall, and word order.
    """
    try:
        # Try to import METEOR from nltk
        from nltk.translate.meteor_score import meteor_score
        from nltk.tokenize import word_tokenize
        import nltk
        
        # Download required NLTK data if not present, with better error handling
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download punkt tokenizer: {e}")
                return get_simple_meteor(generated_text, target_text)
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            try:
                nltk.download('wordnet', quiet=True)
            except Exception as e:
                print(f"Warning: Failed to download wordnet: {e}")
                return get_simple_meteor(generated_text, target_text)
        
        # Tokenize texts
        generated_tokens = word_tokenize(str(generated_text).lower())
        target_tokens = word_tokenize(str(target_text).lower())
        
        # Calculate METEOR score
        # meteor_score expects reference as a list of token lists
        score = meteor_score([target_tokens], generated_tokens)
        return score
        
    except ImportError:
        # Fallback to a simpler implementation if nltk is not available
        return get_simple_meteor(generated_text, target_text)
    except Exception as e:
        # Print warning and fallback if any error occurs
        print(f"Warning: METEOR calculation failed with error: {e}")
        return get_simple_meteor(generated_text, target_text)


def get_simple_meteor(generated_text, target_text):
    """Simple METEOR-like metric when nltk is not available.
    
    This is a simplified version that approximates METEOR using word overlap.
    """
    generated_words = set(str(generated_text).lower().split())
    target_words = set(str(target_text).lower().split())
    
    if len(target_words) == 0:
        return 0.0
    
    # Calculate precision and recall
    common_words = generated_words.intersection(target_words)
    
    if len(generated_words) == 0:
        precision = 0.0
    else:
        precision = len(common_words) / len(generated_words)
    
    recall = len(common_words) / len(target_words)
    
    # F-mean with alpha=0.9 (METEOR uses this weighting)
    if precision + recall == 0:
        return 0.0
    
    f_mean = (10 * precision * recall) / (9 * precision + recall)
    return f_mean


METRIC_FNS = {
    "accuracy": get_accuracy,
    "rouge": get_rouge,  # GPU-accelerated single-pair ROUGE computation
    "rouge_batch": get_rouge_batch,  # GPU-accelerated batch ROUGE computation for better throughput
    "meteor": get_meteor,
    "mae": get_mae,
    "rmse": get_rmse,
    "f1": get_f1_score,
    "stsb": get_stsb,
    "dbpedia": get_dbpedia,
    "drop": get_drop,
    "label_and_explanation": get_label_and_explanation,
    "binary_accuracy_flex": get_binary_accuracy_flex,
    "gsm8k_regex": get_gsm8k_regex,
    "mrpc_accuracy": get_mrpc_accuracy,
    "mnli_accuracy": get_mnli_accuracy,
    "hellaswag_accuracy": get_hellaswag_accuracy,
    "customer_support_accuracy": get_customer_support_accuracy,
    "amazon_review_mpe": get_stsb,  # 1-5 ratings.
}
