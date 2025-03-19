# -*- coding: utf-8 -*-
"""
文本分块和处理工具
"""

from typing import List, Dict
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from src.logger import logger

class TextSplitter:
    """文本分块和处理工具"""
    
    def __init__(self):
        """初始化分块器"""
        self.stop_words = set(stopwords.words('english'))
        self.tfidf = TfidfVectorizer(stop_words='english')
        
    def get_sentence_scores(self, sentences: List[str]) -> Dict[str, float]:
        """计算句子重要性得分"""
        if not sentences:
            return {}
            
        # 确保至少有一个句子
        if len(sentences) == 1:
            return {sentences[0]: 1.0}
        try:
            # TF-IDF 得分
            tfidf_matrix = self.tfidf.fit_transform(sentences)
            sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # 位置得分
            position_scores = [1.0 / (i + 1) for i in range(len(sentences))]
            
            # 组合得分
            final_scores = {}
            for i, sentence in enumerate(sentences):
                score = 0.7 * sentence_scores[i] + 0.3 * position_scores[i]
                final_scores[sentence] = score
                
            return final_scores
        except Exception as e:
            logger.error(f"计算句子得分时出错: {str(e)}")
            return {sentence: 1.0/i for i, sentence in enumerate(sentences, 1)}

    def smart_truncate(self, text: str, max_length: int = 1024) -> str:
        """
        智能截断文本，保留最重要的内容
        使用句子重要性评分和语义连贯性
        """
        if len(text) <= max_length:
            return text
            
        sentences = sent_tokenize(text)
        scores = self.get_sentence_scores(sentences)
        
        # 选择最重要的句子
        selected_sentences = []
        current_length = 0
        
        # 按得分排序句子
        sorted_sentences = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for sentence, _ in sorted_sentences:
            if current_length + len(sentence) + 1 <= max_length:
                selected_sentences.append(sentence)
                current_length += len(sentence) + 1
            else:
                break
                
        # 恢复原始顺序
        original_order = [s for s in sentences if s in selected_sentences]
        return ' '.join(original_order)

    def semantic_chunk(self, text: str, max_length: int = 1024, overlap: int = 100) -> List[str]:
        """
        基于语义的文本分块
        使用句子向量的相似度来确定最佳分块点
        """
        if len(text) <= max_length:
            return [text]
            
        # 分句
        sentences = sent_tokenize(text)
        
        try:
            # 计算句子向量
            sentence_vectors = self.tfidf.fit_transform(sentences).toarray()
            
            # 计算句子间的语义相似度矩阵
            similarity_matrix = np.dot(sentence_vectors, sentence_vectors.T)
            
            chunks = []
            current_chunk = []
            current_length = 0
            last_break_idx = 0
            
            for i, sentence in enumerate(sentences):
                sentence_len = len(sentence)
                
                if current_length + sentence_len > max_length and current_chunk:
                    # 寻找最佳分块点
                    if i > last_break_idx + 1:
                        # 计算当前位置前后的语义连贯性
                        pre_coherence = np.mean(similarity_matrix[i-1, max(0, i-3):i])
                        post_coherence = np.mean(similarity_matrix[i, i+1:min(len(sentences), i+4)])
                        
                        # 如果后续语义连贯性更强，尝试在前一个位置断开
                        if post_coherence > pre_coherence:
                            current_chunk = current_chunk[:-1]
                    
                    chunks.append(' '.join(current_chunk))
                    
                    # 智能选择重叠部分
                    if current_chunk:
                        # 基于语义相似度选择最相关的前几个句子作为重叠
                        overlap_scores = similarity_matrix[i, max(0, i-5):i]
                        best_overlap_idx = np.argsort(overlap_scores)[-2:] # 选择最相关的2个句子
                        overlap_sentences = [sentences[j] for j in best_overlap_idx + max(0, i-5)]
                        current_chunk = overlap_sentences
                        current_length = sum(len(s) for s in current_chunk)
                        last_break_idx = i
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_len
            
            # 处理最后一个块
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            # 优化块之间的连接
            final_chunks = []
            for i in range(len(chunks)):
                if i > 0:
                    # 添加过渡句子
                    prev_last_sent = sent_tokenize(chunks[i-1])[-1]
                    curr_first_sent = sent_tokenize(chunks[i])[0]
                    transition_score = similarity_matrix[
                        sentences.index(prev_last_sent),
                        sentences.index(curr_first_sent)
                    ]
                    
                    # 如果语义断裂较大，添加过渡提示
                    if transition_score < 0.3:
                        chunks[i] = f"[续前文]... {chunks[i]}"
                
                final_chunks.append(chunks[i])
                
            return final_chunks
            
        except Exception as e:
            logger.error(f"语义分块处理失败: {str(e)}")
            # 降级为简单分块
            return self._simple_chunk(text, max_length, overlap)
    
    def _simple_chunk(self, text: str, max_length: int, overlap: int) -> List[str]:
        """简单的文本分块方法"""
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + max_length
            if end >= text_len:
                chunks.append(text[start:])
                break
                
            # 查找合适的断句点
            while end > start and text[end] not in '.!?。！？':
                end -= 1
            if end == start:
                end = start + max_length
                
            chunks.append(text[start:end])
            start = end - overlap
            
        return chunks    

    def extract_key_sections(self, text: str, max_length: int = 1024) -> str:
        """
        智能提取文本关键部分
        使用多种特征进行重要性评估
        """
        if len(text) <= max_length:
            return text
            
        # 分句
        sentences = sent_tokenize(text)
        
        # 计算句子特征
        sentence_scores = {}
        word_freq = Counter(word_tokenize(text.lower()))
        
        for i, sentence in enumerate(sentences):
            words = word_tokenize(sentence.lower())
            
            # 计算多个特征
            features = {
                'position': 1.0 / (i + 1),  # 位置得分
                'length': len(words) / 100,  # 长度得分
                'word_importance': sum(word_freq[word] for word in words) / len(words),  # 词重要性
                'contains_numbers': any(c.isdigit() for c in sentence),  # 是否包含数字
                'contains_names': any(w[0].isupper() for w in words[1:])  # 是否包含专有名词
            }
            
            # 特征加权
            score = (
                0.3 * features['position'] +
                0.2 * features['length'] +
                0.3 * features['word_importance'] +
                0.1 * features['contains_numbers'] +
                0.1 * features['contains_names']
            )
            
            sentence_scores[sentence] = score
        
        # 选择最重要的句子
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 提取关键部分
        selected_text = []
        current_length = 0
        
        for sentence, _ in sorted_sentences:
            if current_length + len(sentence) <= max_length:
                selected_text.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        # 按原始顺序重排
        final_text = []
        for sentence in sentences:
            if sentence in selected_text:
                final_text.append(sentence)
                
        return ' '.join(final_text)
