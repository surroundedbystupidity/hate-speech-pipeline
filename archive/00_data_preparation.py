#!/usr/bin/env python3
"""
Data Preparation Script for Reddit Hate Speech Analysis.
Implements Davidson et al. lexicon-based labeling and balanced dataset creation.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import json
import re
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 添加Transformers库用于预训练模型
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# 简化的二分类lexicon - 只区分hate/not hate
HATE_LEXICON = [
    # 来自用户词汇表的所有词汇 (prophate >= 0.3的词汇)
    'allah akbar', 'all niggers', 'faggots like', 'faggots usually', 'fucking nigger',
    'get raped', 'hate faggots', 'raped by', 'shut up nigger', 'this nigger',
    'you nigger', 'you niggers', 'a fucking queer', 'faggots like you',
    'fucking hate you', 'full of white', 'how many niggers', 'is a fucking',
    'many niggers are', 'niggers are in', 'full of white trash',
    'how many niggers are', 'is full of white', 'many niggers are in',
    'niggers are in my', 'blacks', 'chink', 'chinks', 'dykes', 'faggot', 'faggots', 'fags',
    'homo', 'inbred', 'niggers', 'queers', 'raped', 'savages', 'slave',
    'spic', 'wetback', 'wetbacks', 'whites', 'a dirty', 'a nigger',
    'all white', 'always fuck', 'ass white', 'be killed', 'beat him',
    'biggest faggot', 'blame the', 'butt ugly', 'chink eyed', 'chinks in',
    'coon shit', 'dumb monkey', 'dumb nigger', 'fag and', 'fag but',
    'faggot a', 'faggot and', 'faggot ass', 'faggot bitch', 'faggot for',
    'faggot smh', 'faggot that', 'faggots and', 'fags are', 'fuckin faggot',
    'fucking faggot', 'fucking gay', 'fucking hate', 'fucking queer',
    'gay ass', 'hate all', 'hate fat', 'hate you', 'here faggot',
    'is white', 'jungle bunny', 'kill all', 'kill yourself', 'little faggot',
    'married to', 'me faggot', 'my coon', 'nigga ask', 'niggas like',
    'nigger ass', 'nigger is', 'nigger music', 'niggers are', 'of fags',
    'of white', 'raped and', 'sand nigger', 'savages that', 'shorty bitch',
    'spear chucker', 'spic cop', 'stupid nigger', 'that fag', 'that faggot',
    'that nigger', 'the faggots', 'the female', 'the niggers', 'their heads',
    'them white', 'then faggot', 'to rape', 'trailer park', 'trash with',
    'u fuckin', 'ugly dyke', 'up nigger', 'white ass', 'white boy',
    'white person', 'white trash', 'with niggas', 'you fag', 'your faggot',
    'your nigger', 'a bitch made', 'a fag and', 'a fag but', 'a faggot and', 'a faggot for',
    'a nigga ask', 'a white person', 'a white trash', 'all these fucking',
    'are all white', 'be killed for', 'bitch made nigga', 'fuck outta here',
    'fuck u talking', 'fuck you too', 'him a nigga', 'his shorty bitch',
    'is a fag', 'is a faggot', 'is a fuckin', 'is a nigger', 'like a faggot',
    'like da colored', 'of white trash', 'still a faggot', 'the biggest faggot',
    'the faggots who', 'the fuck do', 'they all look', 'what a fag',
    'white bitch in', 'white trash and', 'you a fag', 'you a lame',
    'you a nigger', 'you fuck wit', 'you fucking faggot', 'your a cunt',
    'your a dirty', 'your bitch in',
    # 原有词汇
    'nigger', 'nigga', 'kike', 'spic', 'wetback', 'chink', 'gook', 'jap',
    'towelhead', 'raghead', 'sandnigger', 'paki', 'curry', 'beaner',
    'slant', 'yellow', 'oriental', 'mick', 'paddy', 'wop', 'dago', 'guinea',
    'kraut', 'hun', 'frog', 'limey', 'faggot', 'fag', 'dyke', 'tranny',
    'homo', 'queer', 'lesbo', 'butch', 'fairy', 'sissy', 'pansy', 'fruit',
    'twink', 'bear', 'trannie', 'bitch', 'slut', 'whore', 'cunt', 'pussy',
    'dick', 'cock', 'asshole', 'bastard', 'cow', 'pig', 'dog', 'skank',
     'ho', 'tramp', 'harlot', 'retard', 'retarded', 'spaz', 'cripple',
     'gimp', 'lame', 'dumb', 'idiot', 'moron', 'imbecile', 'feeble', 'handicapped',

     # 我自己添加的词汇
     # 暴力威胁词汇
     'kill', 'kills', 'killing', 'murder', 'murders', 'murdering', 'violence', 'violent',
     'beat', 'beats', 'beating', 'attack', 'attacks', 'attacking', 'assault', 'assaults',
     'threat', 'threats', 'threatening', 'threaten', 'die', 'dies', 'dying', 'death', 'dead',
     'suicide', 'should die', 'deserve to die', 'need to die', 'want to die', 'hope you die',

     # 仇恨表达词汇
     'hate', 'hates', 'hating', 'hated', 'despise', 'despises', 'despising', 'despised',
     'disgusting', 'disgust', 'disgusts', 'disgusted', 'sick', 'sickening', 'evil', 'devil',
     'monster', 'monsters', 'scum', 'trash', 'garbage', 'vermin', 'pest', 'pests',
     'parasite', 'parasites', 'cancer', 'disease', 'plague',

     # 去人性化词汇
     'animal', 'animals', 'beast', 'beasts', 'barbarian', 'barbarians', 'subhuman',
     'inhuman', 'less than human', 'not human', 'filth', 'filthy',

     # 群体歧视词汇
     'all blacks', 'all whites', 'all muslims', 'all jews', 'all asians', 'all mexicans',
     'all immigrants', 'all foreigners', 'all gays', 'all lesbians', 'all trans',

     # 具体威胁短语
     'go back to', 'return to', 'get out of', 'fuck off', 'fuck you', 'fuck them',
     'burn in hell', 'go to hell', 'rot in hell', 'damn you', 'damned',

     # 宗教仇恨
     'jihad', 'terrorist', 'terrorists', 'terrorism', 'islamic state', 'isis', 'al qaeda',
     'bomb', 'bombs', 'bombing', 'explode', 'explodes', 'explosion',

     # 政治极端主义
     'nazi', 'nazis', 'hitler', 'holocaust', 'genocide', 'slavery', 'lynching',
     'kkk', 'klan', 'white power', 'aryan', 'supremacist', 'supremacists', 'fascist', 'fascists',

     # 更多冒犯性词汇
     'damn', 'hell', 'shit', 'fuck', 'fucking', 'motherfucker', 'fucker',
     'bullshit', 'crap', 'piss', 'pissed', 'bloody', 'bitchy',

     # 网络仇恨用语
     'kys', 'gtfo', 'stfu', 'fuck off', 'eat shit', 'go die', 'kill yourself',
     'you suck', 'you\'re trash', 'you\'re garbage', 'piece of shit', 'worthless',
     'pathetic', 'disgusting', 'revolting', 'vile', 'despicable', 'contemptible',

     # 更多重要的仇恨表达
     'criminal', 'criminals', 'terrorist', 'terrorists', 'mob', 'mobs', 'violent',
     'capitol', 'murder', 'murdering', 'obstruct', 'certification', 'election',
     'senate', 'republicans', 'fear', 'lives', 'day', 'fuck', 'fucking', 'fucked',
     'shit', 'shitty', 'damn', 'damned', 'hell', 'suck', 'sucks', 'sucking',
     'stupid', 'idiot', 'moron', 'dumb', 'retard', 'asshole', 'bastard',
     'hate', 'hated', 'hating', 'despise', 'disgust', 'disgusting', 'sick',
     'evil', 'devil', 'monster', 'scum', 'trash', 'garbage', 'worthless',
     'pathetic', 'revolting', 'vile', 'despicable', 'contemptible', 'awful',
     'terrible', 'horrible', 'disgusting', 'revolting', 'sickening', 'nasty',
     'gross', 'filthy', 'dirty', 'rotten', 'corrupt', 'corrupted', 'criminal',

     # 从验证中发现的真正仇恨词汇
     'mentally challenged', 'narcissism', 'narcissist', 'white america',
     'maga loyalists', 'maga', 'loyalists', 'loyalist', 'woke', 'crazy',
     'insane', 'lunatic', 'psycho', 'psychotic', 'nasty', 'gnat',

     # 更多重要的仇恨词汇
     # 种族和民族歧视
     'white trash', 'black trash', 'yellow peril', 'brown people', 'redskin', 'redskins',
     'wetback', 'beaner', 'spic', 'chink', 'gook', 'jap', 'towelhead', 'raghead',
     'sand nigger', 'sandnigger', 'paki', 'curry muncher', 'camel jockey',

     # 性别和性取向歧视
     'tranny', 'shemale', 'ladyboy', 'fag hag', 'dyke', 'butch', 'femme',
     'sissy', 'pansy', 'fruit', 'twink', 'bear', 'otter', 'chub', 'chaser',
     'gold star', 'pillow princess', 'stone butch', 'lipstick lesbian',

     # 残疾和心理健康歧视
     'retard', 'spaz', 'cripple', 'gimp', 'lame', 'dumbass', 'idiot', 'moron',
     'imbecile', 'feeble', 'handicapped', 'wheelchair bound', 'special needs',
     'autistic', 'aspergers', 'bipolar', 'schizo', 'psycho', 'mental case',

     # 宗教仇恨
     'christ killer', 'jew', 'kike', 'heeb', 'yid', 'jewboy', 'jewess',
     'muzzie', 'raghead', 'towelhead', 'sand monkey', 'camel jockey',
     'allah snackbar', 'jihad', 'terrorist', 'bomber', 'suicide bomber',

     # 政治极端主义
     'commie', 'commie bastard', 'red', 'pinko', 'libtard', 'conservatard',
     'republicunt', 'democrap', 'trumptard', 'biden voter', 'clinton voter',
     'obama voter', 'snowflake', 'snowflake lib', 'libtard', 'conservatard',

     # 网络仇恨用语
     'kys', 'gtfo', 'stfu', 'eat shit', 'go die', 'kill yourself', 'neck yourself',
     'rope', 'hang yourself', 'jump off a bridge', 'drink bleach', 'eat glass',
     'you suck', 'you\'re trash', 'you\'re garbage', 'piece of shit', 'worthless',
     'pathetic', 'disgusting', 'revolting', 'vile', 'despicable', 'contemptible',

     # 威胁和暴力表达
     'i hope you die', 'i wish you were dead', 'you should die', 'deserve to die',
     'need to die', 'want to die', 'hope you die', 'burn in hell', 'go to hell',
     'rot in hell', 'damn you', 'damned', 'cursed', 'doomed', 'fucked',

     # 去人性化表达
     'subhuman', 'inhuman', 'less than human', 'not human', 'filth', 'filthy',
     'vermin', 'pest', 'parasite', 'cancer', 'disease', 'plague', 'infection',
     'contamination', 'pollution', 'toxin', 'poison', 'venom', 'sewage'
]


DAVIDSON_HATE_LEXICON = {
    'hate_terms': HATE_LEXICON,
    # Offensive but not necessarily hate speech
    'offensive_terms': [
        'damn', 'hell', 'shit', 'fuck', 'fucking', 'motherfucker', 'fucker',
        'bullshit', 'crap', 'piss', 'pissed', 'damned', 'bloody', 'bitchy',
        'ass', 'butt', 'tits', 'boobs', 'breast', 'penis', 'vagina',
        'sex', 'sexual', 'porn', 'pornography', 'masturbate', 'orgasm'
    ],
    # Context-dependent terms that can indicate hate speech
    'context_terms': [
        # Violence and death
        'kill', 'die', 'death', 'murder', 'rape', 'violence', 'terrorist',
        'bomb', 'explode', 'shoot', 'stab', 'beat', 'attack', 'assault',
        # Historical hate symbols
        'nazi', 'hitler', 'holocaust', 'genocide', 'slavery', 'lynching',
        'kkk', 'klan', 'white power', 'aryan', 'supremacist', 'fascist',
        # Dehumanizing terms
        'vermin', 'pest', 'parasite', 'cancer', 'disease', 'plague',
        'infest', 'invade', 'flood', 'swarm', 'horde', 'invasion',
        # Political extremism
        'jihad', 'terror', 'extremist', 'radical', 'fundamentalist',
        'islamist', 'militant', 'insurgent', 'rebel', 'traitor'
    ],
    # Additional patterns for hate speech detection
    'hate_patterns': [
        # Group targeting patterns
        r'\b(all|every|most)\s+(muslims|jews|blacks|whites|asians|mexicans|immigrants)\b',
        r'\b(they|them|those)\s+(people|animals|vermin|scum|trash)\b',
        r'\b(go back to|return to|get out of)\s+(your country|africa|mexico|china)\b',
        # Dehumanization patterns
        r'\b(not human|subhuman|less than human|animals|beasts)\b',
        r'\b(deserve to die|should be killed|need to be eliminated)\b',
        # Threat patterns
        r'\b(i will|i\'ll|we will|we\'ll)\s+(kill|hurt|harm|destroy)\b',
        r'\b(you should|they should|we should)\s+(die|be killed|be eliminated)\b'
    ]
}

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def contains_hate_speech(text):
    """
    使用lexicon检查文本是否包含仇恨言论
    Returns: (is_hate, matched_terms)
    - is_hate: True/False
    - matched_terms: 匹配到的词汇列表
    """
    if pd.isna(text) or not text:
        return False, []

    text_lower = str(text).lower()
    matched_terms = []

    # 使用简单的包含匹配，确保高召回率
    for term in HATE_LEXICON:
        if term in text_lower:
            matched_terms.append(term)

    return len(matched_terms) > 0, matched_terms

def load_reddit_data_step1(config):
    """
    Step 1 - Data processing
    Subset the December 1, 2024 comment dataset to first two hours
    """
    print("=== Step 1: Data Processing ===")
    print("Loading December 1, 2024 comment dataset - first 2 hours only")

    time_window_hours = 2  # 2小时时间窗口
    print(f"Time window: {time_window_hours} hours")

    # Load submissions
    mini_dir = Path(config['paths']['mini_dir'])
    submissions_files = list(mini_dir.glob("*submission*.parquet"))

    if not submissions_files:
        print("No submission files found, checking mini_dataset...")
        mini_dir = Path("mini_dataset")
        submissions_files = list(mini_dir.glob("*submission*.parquet"))

    submissions_dfs = []
    for file in submissions_files:  # Process all submission files
        try:
            df = pd.read_parquet(file)
            submissions_dfs.append(df)
            print(f"Loaded {len(df)} submissions from {file.name}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    submissions_df = pd.concat(submissions_dfs, ignore_index=True) if submissions_dfs else pd.DataFrame()

    # Load comments
    comments_files = list(mini_dir.glob("*comment*.parquet"))

    if not comments_files:
        print("No comment files found, checking mini_dataset...")
        mini_dir = Path("mini_dataset")
        comments_files = list(mini_dir.glob("*comment*.parquet"))

    comments_dfs = []
    for file in comments_files:  # Process all comment files
        try:
            df = pd.read_parquet(file)
            print(f"Original file has {len(df)} comments")

            # 应用基于词典的初步过滤来减少数据量
            print("Applying initial hate lexicon filtering...")
            df['text_content'] = df.get('body', '').fillna('')

            # 使用简化词典过滤来减少数据量
            def contains_hate_terms(text):
                is_hate, _ = contains_hate_speech(text)
                return is_hate

            tqdm.pandas(desc="Filtering hate speech")
            hate_mask = df['text_content'].progress_apply(contains_hate_terms)
            hate_comments = df[hate_mask]
            print(f"Found {len(hate_comments)} potential hate comments out of {len(df)} total")

            if len(hate_comments) > 0:
                # 按时间排序，获取2小时时间窗口
                hate_comments = hate_comments.sort_values('created_utc')
                start_time = hate_comments['created_utc'].min()
                end_time = start_time + pd.Timedelta(hours=time_window_hours)

                # 过滤2小时时间窗口内的仇恨评论
                time_filtered = hate_comments[
                    (hate_comments['created_utc'] >= start_time) &
                    (hate_comments['created_utc'] <= end_time)
                ]
                print(f"Found {len(time_filtered)} potential hate comments in 2-hour window")

                # 获取这些评论的完整线程（通过parent_id构建）
                filtered_df = build_complete_threads(time_filtered, df)
                print(f"Built complete threads with {len(filtered_df)} comments")
            else:
                filtered_df = df.head(0)  # 空DataFrame

            # 不再进行采样，保留所有过滤后的数据

            comments_dfs.append(filtered_df)
            print(f"Loaded {len(filtered_df)} comments from {file.name}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    comments_df = pd.concat(comments_dfs, ignore_index=True) if comments_dfs else pd.DataFrame()

    print(f"Total loaded: {len(submissions_df)} submissions and {len(comments_df)} comments")
    return submissions_df, comments_df

class CardiffNLPHateClassifier:
    """Cardiff NLP预训练仇恨言论分类器"""

    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-hate-latest"
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()

    def _load_model(self):
        """加载预训练模型和分词器"""
        print(f"Loading Cardiff NLP model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def classify_text(self, text):
        """
        使用Cardiff NLP模型分类文本
        Returns: 0 = not hate, 1 = hate speech
        """
        if pd.isna(text) or not text or len(text.strip()) == 0:
            return 0

        try:
            # 预处理文本
            text = str(text).strip()
            if len(text) > 512:  # 截断长文本
                text = text[:512]

            # 分词和编码
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()

            # Cardiff NLP模型输出: 0=not hate, 1=hate
            return predicted_class

        except Exception as e:
            print(f"Error classifying text: {e}")
            return 0  # 默认为非仇恨言论

    def classify_batch(self, texts):
        """
        批量分类文本，充分利用GPU
        Returns: list of labels (0 = not hate, 1 = hate)
        """
        if not texts:
            return []

        # 快速预处理所有文本
        processed_texts = []
        for text in texts:
            if pd.isna(text) or not text or len(str(text).strip()) == 0:
                processed_texts.append("")
            else:
                text = str(text).strip()
                if len(text) > 512:
                    text = text[:512]
                processed_texts.append(text)

        try:
            # 批量分词和编码 - 使用更高效的设置
            inputs = self.tokenizer(
                processed_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True
            ).to(self.device, non_blocking=True)

            # 批量推理 - 使用torch.compile加速（如果可用）
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(predictions, dim=-1)

                # 立即移动到CPU释放GPU内存
                predicted_classes = predicted_classes.cpu().numpy()

            return predicted_classes.tolist()

        except Exception as e:
            print(f"Error in batch classification: {e}")
            return [0] * len(texts)  # 返回默认标签

def apply_cardiffnlp_labeling(texts, batch_size=256):
    """
    批量应用Cardiff NLP标注 - 高度优化版本
    Returns: list of labels (0 = not hate, 1 = hate)
    """
    classifier = CardiffNLPHateClassifier()
    labels = []

    print(f"Labeling {len(texts)} texts with Cardiff NLP model (batch_size={batch_size})...")
    print(f"GPU device: {classifier.device}")

    # 预分配结果列表
    labels = [0] * len(texts)

    # 真正的批量处理 - 减少循环开销
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Cardiff NLP labeling"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]

        # 批量推理
        batch_labels = classifier.classify_batch(batch_texts)

        # 直接赋值到预分配列表
        labels[start_idx:end_idx] = batch_labels

    return labels

def create_time_based_split(df):
    """
    创建基于时间的数据分割：验证集(前15分钟)、训练集(中间90分钟)、测试集(最后15分钟)
    """
    print("Creating time-based data split...")

    if df.empty:
        return df, df, df

    # 获取时间范围
    min_time = df['created_utc'].min()
    max_time = df['created_utc'].max()
    total_duration = max_time - min_time

    print(f"Time range: {total_duration.total_seconds() / 3600:.2f} hours")

    # 计算分割点（使用Timedelta）
    validate_end = min_time + pd.Timedelta(minutes=15)  # 前15分钟
    train_end = max_time - pd.Timedelta(minutes=15)     # 最后15分钟之前

    # 分割数据
    validate_df = df[(df['created_utc'] >= min_time) & (df['created_utc'] <= validate_end)]
    train_df = df[(df['created_utc'] > validate_end) & (df['created_utc'] <= train_end)]
    test_df = df[df['created_utc'] > train_end]

    print(f"Data split:")
    print(f"  Validate: {len(validate_df)} samples ({len(validate_df)/len(df)*100:.1f}%)")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

    return validate_df, train_df, test_df

def human_verify_sample(df, sample_size=100, random_state=42):
    """
    人工验证标注样本的准确性
    """
    print(f"Human verification of {sample_size} samples...")

    if len(df) < sample_size:
        sample_df = df.copy()
    else:
        sample_df = df.sample(n=sample_size, random_state=random_state)

    verification_results = []

    print("Please verify the following labels (press Enter to continue, 'q' to quit):")
    print("Format: [Index] Text -> Predicted Label")
    print("=" * 80)

    for idx, row in sample_df.iterrows():
        text = str(row['text_content'])[:200] + "..." if len(str(row['text_content'])) > 200 else str(row['text_content'])
        predicted_label = row.get('cardiffnlp_label', row.get('davidson_label', 0))

        print(f"[{idx}] {text}")
        print(f"Predicted: {'HATE' if predicted_label == 1 else 'NOT HATE'}")

        user_input = input("Correct? (y/n/q): ").strip().lower()

        if user_input == 'q':
            break
        elif user_input == 'n':
            verification_results.append({'index': idx, 'predicted': predicted_label, 'correct': False})
        else:
            verification_results.append({'index': idx, 'predicted': predicted_label, 'correct': True})

        print("-" * 80)

    if verification_results:
        accuracy = sum(1 for r in verification_results if r['correct']) / len(verification_results)
        print(f"\nHuman verification accuracy: {accuracy:.2%} ({len(verification_results)} samples)")
    else:
        print("No verification performed.")

    return verification_results

def find_hate_threads_cardiffnlp(thread_data, comments_df):
    """
    使用Cardiff NLP标签识别仇恨线程
    """
    print("Finding hate threads using Cardiff NLP labels...")

    hate_threads = []

    for link_id, thread_info in tqdm(thread_data.items(), desc="Checking threads for hate"):
        # 获取该线程的所有评论
        thread_comments = comments_df[comments_df['link_id'] == link_id]

        # 检查是否包含仇恨言论 (Cardiff NLP标签为1)
        if (thread_comments['cardiffnlp_label'] == 1).any():
            # 计算线程统计信息
            hate_thread = {
                'link_id': link_id,
                'subreddit': thread_comments.iloc[0]['subreddit'],
                'total_comments': len(thread_comments),
                'unique_authors': thread_comments['author'].nunique(),
                'time_span': thread_comments['created_utc'].max() - thread_comments['created_utc'].min(),
                'hate_comments_count': (thread_comments['cardiffnlp_label'] == 1).sum(),
                'root_comment_time': thread_comments['created_utc'].min()
            }
            hate_threads.append(hate_thread)

    print(f"Found {len(hate_threads)} hate threads")
    return hate_threads

def find_matching_non_hate_threads_cardiffnlp(hate_threads, thread_data, comments_df, time_window_hours=2, max_pairs=500):
    """
    为每个仇恨线程找到结构相似的非仇恨线程 (使用Cardiff NLP标签)
    """
    print("Finding matching non-hate threads using Cardiff NLP labels...")

    matched_pairs = []

    # 限制处理的仇恨线程数量
    hate_threads_sample = hate_threads[:max_pairs] if len(hate_threads) > max_pairs else hate_threads
    print(f"Processing {len(hate_threads_sample)} hate threads")

    # 预计算非仇恨线程
    non_hate_threads = []
    for link_id, thread_info in thread_data.items():
        thread_comments = comments_df[comments_df['link_id'] == link_id]
        if not (thread_comments['cardiffnlp_label'] == 1).any():  # 使用Cardiff NLP标签
            non_hate_threads.append({
                'link_id': link_id,
                'thread_info': thread_info,
                'subreddit': thread_comments.iloc[0]['subreddit'] if not thread_comments.empty else '',
                'root_comment_time': thread_comments['created_utc'].min() if not thread_comments.empty else 0
            })

    print(f"Found {len(non_hate_threads)} non-hate threads to match against")

    # 为每个仇恨线程寻找匹配
    for hate_thread in tqdm(hate_threads_sample, desc="Matching threads"):
        best_match = None
        best_similarity = 0.0

        # 在同一subreddit中寻找
        subreddit_threads = [t for t in non_hate_threads if t['subreddit'] == hate_thread['subreddit']]

        for non_hate_thread in subreddit_threads:
            # 时间窗口检查
            time_diff = abs(hate_thread['root_comment_time'] - non_hate_thread['root_comment_time'])
            if time_diff > pd.Timedelta(hours=time_window_hours):
                continue

            # 计算相似度
            similarity_score = calculate_thread_similarity(hate_thread, non_hate_thread['thread_info'])

            if similarity_score > best_similarity and similarity_score > 0.1:
                best_similarity = similarity_score
                best_match = non_hate_thread

        if best_match:
            matched_pairs.append({
                'hate_thread': hate_thread,
                'non_hate_thread': best_match,
                'similarity_score': best_similarity
            })

    print(f"Successfully matched {len(matched_pairs)} thread pairs")
    return matched_pairs


def build_complete_threads(hate_comments, all_comments):
    """
    通过parent_id构建完整的评论线程
    包括仇恨评论的所有父评论和子评论
    """
    print("Building complete threads from hate comments...")

    if hate_comments.empty:
        return pd.DataFrame()

    # 收集所有相关的评论ID
    related_comment_ids = set()

    for _, hate_comment in hate_comments.iterrows():
        hate_comment_id = hate_comment['id']
        parent_id = hate_comment['parent_id']
        link_id = hate_comment['link_id']

        # 添加仇恨评论本身
        related_comment_ids.add(hate_comment_id)

        # 向上追溯所有父评论
        current_parent = parent_id
        while current_parent and current_parent != link_id:
            # 查找父评论
            parent_comment = all_comments[all_comments['id'] == current_parent]
            if not parent_comment.empty:
                related_comment_ids.add(current_parent)
                current_parent = parent_comment.iloc[0]['parent_id']
            else:
                break

        # 向下收集所有子评论（递归查找）
        def collect_children(comment_id):
            children = all_comments[all_comments['parent_id'] == comment_id]
            for _, child in children.iterrows():
                related_comment_ids.add(child['id'])
                collect_children(child['id'])  # 递归查找子评论的子评论

        collect_children(hate_comment_id)

    # 获取所有相关评论
    related_comments = all_comments[all_comments['id'].isin(related_comment_ids)]

    print(f"Built complete threads: {len(related_comment_ids)} related comments from {len(hate_comments)} hate comments")

    return related_comments

def extract_comment_threads(comments_df):
    """
    提取完整的评论线程结构
    返回: 每个线程的完整评论树
    """
    print("Extracting comment threads...")

    # 按link_id分组，每个link_id代表一个帖子下的所有评论
    thread_data = {}

    for link_id, group in tqdm(comments_df.groupby('link_id'), desc="Processing threads"):
        if pd.isna(link_id):
            continue

        # 构建评论树结构
        comments_dict = {}
        root_comments = []

        # 首先收集所有评论
        for _, comment in group.iterrows():
            comments_dict[comment['id']] = {
                'id': comment['id'],
                'parent_id': comment['parent_id'],
                'author': comment['author'],
                'body': comment['body'],
                'created_utc': comment['created_utc'],
                'score': comment['score'],
                'subreddit': comment['subreddit'],
                'children': []
            }

        # 构建父子关系
        for comment_id, comment_data in comments_dict.items():
            parent_id = comment_data['parent_id']

            # 如果是根评论（直接回复帖子）
            if parent_id == link_id:
                root_comments.append(comment_id)
            # 如果是回复其他评论
            elif parent_id in comments_dict:
                comments_dict[parent_id]['children'].append(comment_id)

        # 计算线程统计信息
        thread_stats = {
            'link_id': link_id,
            'subreddit': group['subreddit'].iloc[0],
            'total_comments': len(group),
            'unique_authors': group['author'].nunique(),
            'time_span': group['created_utc'].max() - group['created_utc'].min(),
            'root_comments': len(root_comments),
            'comments_dict': comments_dict,
            'root_comment_ids': root_comments
        }

        thread_data[link_id] = thread_stats

    print(f"Extracted {len(thread_data)} comment threads")
    return thread_data

def find_hate_threads(thread_data, comments_df):
    """
    识别包含仇恨言论的线程
    """
    print("Identifying hate threads...")

    hate_threads = []

    for link_id, thread_info in thread_data.items():
        # 获取该线程的所有评论
        thread_comments = comments_df[comments_df['link_id'] == link_id]

        # 检查是否有仇恨言论
        hate_comments = thread_comments[thread_comments['davidson_label'] == 2]

        if len(hate_comments) > 0:
            # 找到仇恨评论的完整线程路径
            hate_comment_ids = hate_comments['id'].tolist()

            for hate_comment_id in hate_comment_ids:
                # 提取从根评论到仇恨评论的完整路径
                thread_path = extract_thread_path(hate_comment_id, thread_info['comments_dict'])

                if thread_path:
                    hate_threads.append({
                        'link_id': link_id,
                        'hate_comment_id': hate_comment_id,
                        'thread_path': thread_path,
                        'subreddit': thread_info['subreddit'],
                        'time_span': thread_info['time_span'],
                        'total_comments': thread_info['total_comments'],
                        'unique_authors': thread_info['unique_authors']
                    })

    print(f"Found {len(hate_threads)} hate threads")
    return hate_threads

def extract_thread_path(hate_comment_id, comments_dict):
    """
    提取从根评论到仇恨评论的完整路径
    """
    if hate_comment_id not in comments_dict:
        return None

    path = []
    current_id = hate_comment_id

    # 向上追溯到根评论
    while current_id in comments_dict:
        path.insert(0, comments_dict[current_id])
        parent_id = comments_dict[current_id]['parent_id']

        # 如果父ID是link_id，说明到达了根评论
        if parent_id.startswith('t3_'):
            break
        current_id = parent_id

    # 向下收集所有子评论
    def collect_children(comment_id):
        if comment_id in comments_dict:
            comment_data = comments_dict[comment_id].copy()
            comment_data['children_data'] = []
            for child_id in comment_data['children']:
                comment_data['children_data'].append(collect_children(child_id))
            return comment_data
        return None

    # 从根评论开始收集完整树
    if path:
        root_comment = path[0]
        full_thread = collect_children(root_comment['id'])
        return full_thread

    return None

def find_matching_non_hate_threads(hate_threads, thread_data, comments_df, time_window_hours=48, max_pairs=2000):
    """
    为每个仇恨线程找到结构相似的非仇恨线程 (优化版本)
    """
    print("Finding matching non-hate threads (optimized)...")

    matched_pairs = []

    # 限制处理的仇恨线程数量以提高速度
    hate_threads_sample = hate_threads[:max_pairs] if len(hate_threads) > max_pairs else hate_threads
    print(f"Processing {len(hate_threads_sample)} hate threads (out of {len(hate_threads)} total)")

    # 预计算非仇恨线程的统计信息
    non_hate_threads = []
    for link_id, thread_info in thread_data.items():
        # 快速检查是否包含仇恨言论
        thread_comments = comments_df[comments_df['link_id'] == link_id]
        if not (thread_comments['davidson_label'] == 2).any():
            non_hate_threads.append({
                'link_id': link_id,
                'thread_info': thread_info,
                'subreddit': thread_info['subreddit'],
                'time': thread_comments['created_utc'].min()
            })

    print(f"Found {len(non_hate_threads)} non-hate threads for matching")

    for hate_thread in tqdm(hate_threads_sample, desc="Matching threads"):
        hate_link_id = hate_thread['link_id']
        hate_subreddit = hate_thread['subreddit']
        hate_time = comments_df[comments_df['link_id'] == hate_link_id]['created_utc'].min()
        hate_stats = {
            'total_comments': hate_thread['total_comments'],
            'unique_authors': hate_thread['unique_authors'],
            'time_span': hate_thread['time_span']
        }

        # 寻找匹配的非仇恨线程
        best_match = None
        best_score = -1

        # 只检查同一subreddit的非仇恨线程
        same_subreddit_threads = [t for t in non_hate_threads if t['subreddit'] == hate_subreddit]

        for non_hate_thread in same_subreddit_threads:
            link_id = non_hate_thread['link_id']

            # 跳过仇恨线程本身
            if link_id == hate_link_id:
                continue

            # 检查时间窗口
            time_diff = abs(non_hate_thread['time'] - hate_time)
            if time_diff > pd.Timedelta(hours=time_window_hours):  # 转换为秒
                continue

            # 计算结构相似度
            similarity_score = calculate_thread_similarity(hate_stats, {
                'total_comments': non_hate_thread['thread_info']['total_comments'],
                'unique_authors': non_hate_thread['thread_info']['unique_authors'],
                'time_span': non_hate_thread['thread_info']['time_span']
            })

            if similarity_score > best_score:
                best_score = similarity_score
                best_match = {
                    'link_id': link_id,
                    'thread_info': non_hate_thread['thread_info'],
                    'similarity_score': similarity_score
                }

        if best_match and best_score > 0.1:  # 进一步降低相似度阈值以获得更多配对
            matched_pairs.append({
                'hate_thread': hate_thread,
                'non_hate_thread': best_match,
                'similarity_score': best_score
            })

    print(f"Found {len(matched_pairs)} matched thread pairs")
    return matched_pairs

def calculate_thread_similarity(hate_stats, non_hate_stats):
    """
    计算两个线程的结构相似度
    """
    # 归一化差异
    comment_diff = abs(hate_stats['total_comments'] - non_hate_stats['total_comments']) / max(hate_stats['total_comments'], non_hate_stats['total_comments'], 1)
    author_diff = abs(hate_stats['unique_authors'] - non_hate_stats['unique_authors']) / max(hate_stats['unique_authors'], non_hate_stats['unique_authors'], 1    # 处理时间跨度比较
    hate_time_span = hate_stats['time_span']
    non_hate_time_span = non_hate_stats['time_span']

    # 转换为秒进行比较
    if hasattr(hate_time_span, 'total_seconds'):
        hate_time_span = hate_time_span.total_seconds()
    if hasattr(non_hate_time_span, 'total_seconds'):
        non_hate_time_span = non_hate_time_span.total_seconds()

    time_diff = abs(hate_time_span - non_hate_time_span) / max(hate_time_span, non_hate_time_span, 1)

    # 计算综合相似度（越小越相似）
    similarity = 1 - (comment_diff + author_diff + time_diff) / 3
    return max(0, similarity)


# 删除不需要的函数 - 只保留Step 1核心功能

def create_thread_pairing_dataset(matched_pairs, comments_df, config):
    """
    基于1:1配对创建新的数据集
    """
    print("Creating thread pairing dataset...")

    if not matched_pairs:
        print("No matched pairs found!")
        return pd.DataFrame()

    paired_data = []

    for pair in matched_pairs:
        hate_thread = pair['hate_thread']
        non_hate_thread = pair['non_hate_thread']
        similarity_score = pair['similarity_score']

        # 获取仇恨线程的所有评论
        hate_comments = comments_df[comments_df['link_id'] == hate_thread['link_id']].copy()
        hate_comments['thread_type'] = 'hate'
        hate_comments['pair_id'] = f"pair_{hate_thread['link_id']}"
        hate_comments['similarity_score'] = similarity_score

        # 获取非仇恨线程的所有评论
        non_hate_comments = comments_df[comments_df['link_id'] == non_hate_thread['link_id']].copy()
        non_hate_comments['thread_type'] = 'non_hate'
        non_hate_comments['pair_id'] = f"pair_{hate_thread['link_id']}"
        non_hate_comments['similarity_score'] = similarity_score

        paired_data.extend([hate_comments, non_hate_comments])

    if paired_data:
        paired_df = pd.concat(paired_data, ignore_index=True)
        print(f"Created paired dataset with {len(paired_df)} comments from {len(matched_pairs)} pairs")
        return paired_df
    else:
        return pd.DataFrame()

def save_thread_pairing_data(matched_pairs, paired_df, config):
    """
    保存线程配对数据
    """
    print("Saving thread pairing data...")

    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)

    # 保存配对信息
    if matched_pairs:
        pairing_info = []
        for pair in matched_pairs:
            pairing_info.append({
                'pair_id': f"pair_{pair['hate_thread']['link_id']}",
                'hate_link_id': pair['hate_thread']['link_id'],
                'non_hate_link_id': pair['non_hate_thread']['link_id'],
                'subreddit': pair['hate_thread']['subreddit'],
                'similarity_score': pair['similarity_score'],
                'hate_thread_stats': {
                    'total_comments': pair['hate_thread']['total_comments'],
                    'unique_authors': pair['hate_thread']['unique_authors'],
                    'time_span': pair['hate_thread']['time_span']
                },
                'non_hate_thread_stats': {
                    'total_comments': pair['non_hate_thread']['thread_info']['total_comments'],
                    'unique_authors': pair['non_hate_thread']['thread_info']['unique_authors'],
                    'time_span': pair['non_hate_thread']['thread_info']['time_span']
                }
            })

        pairing_path = artifacts_dir / 'thread_pairing_info.json'
        with open(pairing_path, 'w', encoding='utf-8') as f:
            json.dump(pairing_info, f, indent=2, default=str)
        print(f"Thread pairing info saved to {pairing_path}")

    # 保存配对数据集
    if not paired_df.empty:
        paired_data_path = artifacts_dir / 'thread_paired_dataset.parquet'
        paired_df.to_parquet(paired_data_path, engine='pyarrow')
        print(f"Thread paired dataset saved to {paired_data_path}")

def main():
    print("=== Starting main function ===")
    parser = argparse.ArgumentParser(description="Prepare Reddit data with Davidson labeling and thread pairing")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    args = parser.parse_args()
    print(f"Arguments parsed: {args.config}")

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    print("Configuration loaded successfully")

    print("=== Reddit Data Preparation with Cardiff NLP Labeling and Thread Pairing ===")

    # Load Reddit data - Step 1
    submissions_df, comments_df = load_reddit_data_step1(config)

    if submissions_df.empty and comments_df.empty:
        print("No data found! Please check your data paths.")
        return

    # Step 1: 只处理comments，忽略submissions
    print("\n=== Step 1: Processing Comments Only ===")

    if comments_df.empty:
        print("No comments found! Please check your data paths.")
        return

    # 准备文本数据 - 只处理comments
    all_texts = comments_df['text_content'].tolist()

    # Step 1: 使用Cardiff NLP标注所有comments
    print("\n=== Step 1: Cardiff NLP Labeling ===")
    print("Label all comments with cardiffnlp/twitter-roberta-base-hate-latest as hate or not hate")

    # 使用Cardiff NLP模型进行标注
    cardiffnlp_labels = apply_cardiffnlp_labeling(all_texts, batch_size=256)

    # 将标签分配回comments数据
    comments_df['cardiffnlp_label'] = cardiffnlp_labels
    comments_df['source'] = 'comment'

    print(f"Labeled {len(comments_df)} comments")
    hate_count = sum(cardiffnlp_labels)
    print(f"Hate speech comments: {hate_count} ({hate_count/len(comments_df)*100:.1f}%)")
    print(f"Normal comments: {len(comments_df)-hate_count} ({(len(comments_df)-hate_count)/len(comments_df)*100:.1f}%)")

    # Step 1: 跳过人工验证（自动化处理）
    print("\n=== Step 1: Skipping Human Verification (Automated Processing) ===")
    verification_results = []
    print("Skipping human verification for automated processing...")

    # Step 1: 创建基于时间的数据分割
    print("\n=== Step 1: Time-based Data Split ===")
    print("Dataset split:")
    print("  Test – final 15 minutes")
    print("  Train – middle 90 minutes")
    print("  Validate – first 15 minutes")

    # 只使用comments数据
    combined_df = comments_df.copy()
    print(f"Total comments dataset: {len(combined_df)} samples")

    # 创建基于时间的数据分割
    validate_df, train_df, test_df = create_time_based_split(combined_df)

    # 保存分割后的数据
    artifacts_dir = Path(config['paths']['artifacts_dir'])
    artifacts_dir.mkdir(exist_ok=True)

    validate_df.to_parquet(artifacts_dir / 'validate_dataset.parquet', engine='pyarrow')
    train_df.to_parquet(artifacts_dir / 'train_dataset.parquet', engine='pyarrow')
    test_df.to_parquet(artifacts_dir / 'test_dataset.parquet', engine='pyarrow')

    print(f"Time-based splits saved to {artifacts_dir}")

    # Step 1: 线程准备
    print("\n=== Step 1: Thread Preparation ===")
    print("Within each subreddit, find hate speech comments.")
    print("Once found find all parent comments and reply comments.")
    print("This is one hate thread.")
    print("Find within the same subreddit a thread of similar size that does not contain hate speech.")
    print("Root comment should be within the same timestamp as root comment of hate thread.")

    # 提取评论线程
    thread_data = extract_comment_threads(comments_df)

    # 识别仇恨线程 (使用Cardiff NLP标签)
    hate_threads = find_hate_threads_cardiffnlp(thread_data, comments_df)

    # 寻找匹配的非仇恨线程
    matched_pairs = find_matching_non_hate_threads_cardiffnlp(hate_threads, thread_data, comments_df)

    # 创建配对数据集
    paired_df = create_thread_pairing_dataset(matched_pairs, comments_df, config)

    # 保存线程配对数据
    save_thread_pairing_data(matched_pairs, paired_df, config)

    # 保存Step 1的核心数据
    print("\n=== Saving Step 1 Data ===")

    # 保存完整标注的评论数据
    full_data_path = artifacts_dir / 'step1_labeled_comments.parquet'
    combined_df.to_parquet(full_data_path, engine='pyarrow')
    print(f"Step 1 labeled comments saved to {full_data_path}")

    # 保存线程配对信息
    if matched_pairs:
        pairing_path = artifacts_dir / 'step1_thread_pairs.json'
        with open(pairing_path, 'w', encoding='utf-8') as f:
            json.dump(matched_pairs, f, indent=2, default=str)
        print(f"Thread pairs saved to {pairing_path}")

    print(f"\n=== Step 1 Summary ===")
    print(f"Total comments processed: {len(combined_df)} samples")
    print(f"Time-based splits:")
    print(f"  Validate (first 15 min): {len(validate_df)} samples")
    print(f"  Train (middle 90 min): {len(train_df)} samples")
    print(f"  Test (final 15 min): {len(test_df)} samples")

    print(f"\n=== Thread Preparation Summary ===")
    print(f"Total threads analyzed: {len(thread_data)}")
    print(f"Hate threads found: {len(hate_threads)}")
    print(f"Matched thread pairs: {len(matched_pairs)}")
    print(f"Paired dataset size: {len(paired_df)}")

    print("\nStep 1 - Data processing completed successfully!")
    print("Ready for Step 2 - Feature extraction and model training")

if __name__ == "__main__":
    main()
