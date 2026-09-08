"""
Text Prompter Module for MFGF TVPR (Thuần Tiếng Việt - 100% Vietnamese)
Trích xuất các mảnh từ khóa ngữ nghĩa (Danh từ, Động từ, Tính từ) bằng PyVi,
tính toán trọng số tần suất cố định W, và sinh biểu diễn Guided Tips:
    f_tips = W_h ⊙ f^O_tips
"""

import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from pyvi import ViTokenizer, ViPosTagger
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

# Danh sách từ dừng tiếng Việt phổ biến trong mô tả ngoại hình / hành động TVPR
STOPWORDS_VI = {
    "và", "của", "đang", "ở", "với", "có", "một", "các", "những", "cho",
    "vào", "ra", "được", "bị", "để", "như", "thì", "mà", "là", "rất",
    "khi", "trong", "ngoài", "trên", "dưới", "sau", "trước", "cùng",
    "đã", "sẽ", "cũng", "vẫn", "lại", "này", "đó", "kia", "nào", "anh", "cô",
    "ấy", "ta", "họ", "mình", "người_này", "vừa", "rồi", "lúc"
}


class TextPrompter(nn.Module):
    """
    TextPrompter chuyên biệt Tiếng Việt:
      1. Lọc từ khóa ngữ nghĩa (Danh từ, Động từ, Tính từ) bằng PyVi POS Tagging.
      2. Xây dựng ngân hàng từ vựng Tips từ kho ngữ liệu tiếng Việt.
      3. Tính toán véc-tơ trọng số chuẩn hóa W = c_n / C (không huấn luyện / non-trainable).
      4. Sinh véc-tơ Tips thông qua tích Hadamard: f_tips = W_h ⊙ f^O_tips.
    """
    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        word_counts: Optional[Dict[str, int]] = None,
        vocab_size: int = 1000,
        language: str = "vi" # Giữ tham số để tương thích ngược
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.language = "vi"

        if vocab is not None and word_counts is not None:
            self.build_vocab_from_counts(vocab, word_counts)
        else:
            self.register_buffer("W", torch.ones(vocab_size) / vocab_size)
            self.word2idx: Dict[str, int] = {}
            self.idx2word: Dict[int, str] = {}

    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Trích xuất Danh từ (N, Np, Nc), Động từ (V) và Tính từ (A) từ văn bản tiếng Việt.
        """
        text = text.strip()
        keywords = []

        if HAS_PYVI:
            try:
                tokens = ViTokenizer.tokenize(text)
                words, tags = ViPosTagger.postagging(tokens)
                for word, tag in zip(words, tags):
                    clean_w = word.lower().strip(".,;:!?\"'()[]{}")
                    # Lọc Danh từ (N*), Động từ (V*), Tính từ (A*)
                    if (tag.startswith("N") or tag.startswith("V") or tag.startswith("A")) and len(clean_w) > 1:
                        if clean_w not in STOPWORDS_VI:
                            keywords.append(clean_w)
                if keywords:
                    return keywords
            except Exception:
                pass

        # Phân tách từ dự phòng nếu chưa có hoặc lỗi PyVi
        for w in text.lower().split():
            clean_w = w.strip(".,;:!?\"'()")
            if len(clean_w) > 1 and clean_w not in STOPWORDS_VI:
                keywords.append(clean_w)
        return keywords

    def build_vocab_from_corpus(self, text_corpus: List[str], max_vocab_size: Optional[int] = None):
        """
        Xây dựng từ điển từ khóa tiếng Việt và tính toán véc-tơ trọng số hằng số W = c_n / C.
        """
        if max_vocab_size is not None:
            self.vocab_size = max_vocab_size

        keyword_counter = Counter()
        for text in text_corpus:
            kws = self.extract_keywords_from_text(text)
            keyword_counter.update(kws)

        most_common = keyword_counter.most_common(self.vocab_size)
        if not most_common:
            most_common = [("người", 1), ("áo", 1), ("quần", 1), ("đi_bộ", 1)]

        vocab = [word for word, _ in most_common]
        counts = {word: count for word, count in most_common}
        self.build_vocab_from_counts(vocab, counts)

    def build_vocab_from_counts(self, vocab: List[str], counts: Dict[str, int]):
        """
        Tính trọng số non-trainable w_n = c_n / C và đăng ký buffer W.
        """
        self.vocab = vocab[:self.vocab_size]
        self.word2idx = {w: idx for idx, w in enumerate(self.vocab)}
        self.idx2word = {idx: w for idx, w in enumerate(self.vocab)}

        C = sum(counts[w] for w in self.vocab)
        if C == 0:
            C = 1.0

        weights = torch.tensor([counts[w] / C for w in self.vocab], dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-8)

        device = self.W.device if hasattr(self, "W") else torch.device("cpu")
        padded_weights = torch.zeros(self.vocab_size, dtype=torch.float32, device=device)
        padded_weights[:len(weights)] = weights.to(device)

        self.register_buffer("W", padded_weights)

    def get_tips_one_hot(self, texts: Union[str, List[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Tạo véc-tơ hiện diện f^O_tips (multi-hot) cho các câu văn bản tiếng Việt.
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = len(texts)
        f_O_tips = torch.zeros(batch_size, len(self.W), dtype=torch.float32)

        for b_idx, text in enumerate(texts):
            kws = self.extract_keywords_from_text(text)
            for kw in kws:
                if kw in self.word2idx:
                    f_O_tips[b_idx, self.word2idx[kw]] = 1.0

        if device is not None:
            f_O_tips = f_O_tips.to(device)
        return f_O_tips

    def forward(self, texts: Union[str, List[str]], f_O_tips: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Tính biểu diễn Guided Tips: f_tips = W_h ⊙ f^O_tips
        """
        device = self.W.device
        if f_O_tips is None:
            f_O_tips = self.get_tips_one_hot(texts, device=device)
        else:
            f_O_tips = f_O_tips.to(device)

        W_broadcast = self.W.unsqueeze(0)
        f_tips = W_broadcast * f_O_tips
        return f_tips
