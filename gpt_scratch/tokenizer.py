from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent

class BasicTokenizer:

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")

    

tokenizer = BasicTokenizer()

text = "Hello, GPT! ❤️"

ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

# print(ids)
# print(decoded)
# print(text == decoded)
# [72, 101, 108, 108, 111, 44, 32, 71, 80, 84, 33, 32, 226, 157, 164, 239, 184, 143]
# Hello, GPT! ❤️
# True

# print(list("🚀".encode("utf-8")))



def get_stats(ids):

    counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts


def merge(ids, pair, idx):

    newids = []

    i = 0

    while i < len(ids):

        if (
            i < len(ids) - 1
            and ids[i] == pair[0]
            and ids[i + 1] == pair[1]
        ):
            newids.append(idx)
            i += 2

        else:
            newids.append(ids[i])
            i += 1

    return newids

ids = [1, 2, 1, 2, 3, 1, 2]

stats = get_stats(ids)

# print(stats)

ids = merge(ids, (1, 2), 256)

# print(ids)


# [240, 159, 154, 128]
# {(1, 2): 3, (2, 1): 1, (2, 3): 1, (3, 1): 1}
# [256, 256, 3, 256]


class BPETokenizer:

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size):

        assert vocab_size >= 256

        num_merges = vocab_size - 256

        ids = list(text.encode("utf-8"))

        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }

        for i in range(num_merges):

            stats = get_stats(ids)

            if not stats:
                break

            pair = max(
                stats,
                key=stats.get
            )

            idx = 256 + i

            ids = merge(
                ids,
                pair,
                idx
            )

            self.merges[pair] = idx

            self.vocab[idx] = (
                self.vocab[pair[0]]
                + self.vocab[pair[1]]
            )

    
    def encode(self, text):

        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:

            stats = get_stats(ids)

            pairs = [
                pair
                for pair in stats
                if pair in self.merges
            ]

            if not pairs:
                break

            pair = min(
                pairs,
                key=lambda p: self.merges[p]
            )

            idx = self.merges[pair]

            ids = merge(
                ids,
                pair,
                idx
            )

        return ids

    def decode(self, ids):

        tokens = [
            self.vocab[idx]
            for idx in ids
        ]

        text_bytes = b"".join(tokens)

        return text_bytes.decode(
            "utf-8",
            errors="replace"
        )

    def save(self, path):

        data = {
            "merges": [
                [list(pair), idx]
                for pair, idx in self.merges.items()
            ]
        }

        with open(path, "w") as f:
            json.dump(data, f)


    def load(self, path):

        with open(path, "r") as f:
            data = json.load(f)

        self.merges = {
            tuple(pair): idx
            for pair, idx in data["merges"]
        }

        self.vocab = {
            i: bytes([i])
            for i in range(256)
        }

        for pair, idx in self.merges.items():

            self.vocab[idx] = (
                self.vocab[pair[0]]
                +
                self.vocab[pair[1]]
            )



with open(BASE_DIR / "input.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = BPETokenizer()

tokenizer.train(
    text,
    vocab_size=512
)

tokenizer.save(
    BASE_DIR / "tokenizer.json"
)

print("Tokenizer saved to tokenizer.json")

# tokenizer.load(
#    BASE_DIR /  "tokenizer.json"
# )

print(
    "learned merges:",
    len(tokenizer.merges)
)
# learned merges: 256


encoded = tokenizer.encode(
    "hello world"
)

# print(encoded)
# [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]




text = "Hello world! This is GPT."

ids = tokenizer.encode(text)

decoded = tokenizer.decode(ids)

print(ids)
print(decoded)

print(decoded == text)
# [72, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100, 33, 32, 84, 104, 105, 115, 32, 105, 115, 32, 71, 80, 84, 46]
# Hello world! This is GPT.
# True

new_tokenizer = BPETokenizer()

new_tokenizer.load(
   BASE_DIR /  "tokenizer.json"
)

text = "Hello GPT!"

ids = new_tokenizer.encode(text)

print(
    new_tokenizer.decode(ids)
)

# Hello GPT!


test_strings = [
    "",
    "hello world",
    "Hello, GPT!",
    "123456789",
    "new\nline",
    "tabs\ttoo",
    "Unicode: café",
    "Emoji: 🚀🔥",
    "The quick brown fox jumps over the lazy dog."
]

for text in test_strings:

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    assert decoded == text, (
        f"round-trip failed: {text!r}"
    )

print("All tokenizer round-trip tests passed!")


raw_bytes = len(text.encode("utf-8"))
token_count = len(tokenizer.encode(text))

print("raw bytes:", raw_bytes)
print("tokens:", token_count)
print("bytes/token:", raw_bytes / token_count)

# raw bytes: 44
# tokens: 26
# bytes/token: 1.6923076923076923


samples = [
    "こんにちは",
    "你好",
    "नमस्ते",
    "🚀🔥🧠",
    "café résumé naïve"
]

for text in samples:

    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print(text)
    print(ids)
    print(decoded)
    print(text == decoded)
    print()
    
# こんにちは
# [227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175]
# こんにちは
# True

# 你好
# [228, 189, 160, 229, 165, 189]
# 你好
# True

# नमस्ते
# [224, 164, 168, 224, 164, 174, 224, 164, 184, 224, 165, 141, 224, 164, 164, 224, 165, 135]
# नमस्ते
# True

# 🚀🔥🧠
# [240, 159, 154, 128, 240, 159, 148, 165, 240, 159, 167, 160]
# 🚀🔥🧠
# True

# café résumé naïve
# [454, 102, 195, 169, 32, 114, 195, 169, 489, 109, 195, 169, 32, 110, 97, 195, 175, 118, 101]
# café résumé naïve
# True