from pathlib import Path

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

# with open(BASE_DIR / "input.txt", "r", encoding="utf-8") as f:
#     text = f.read()

# tokenizer = BPETokenizer()

# tokenizer.train(
#     text,
#     vocab_size=512
# )

# print(
#     "learned merges:",
#     len(tokenizer.merges)
# )
# learned merges: 256


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

encoded = tokenizer.encode(
    "hello world"
)

print(encoded)
# [104, 101, 108, 108, 111, 32, 119, 111, 114, 108, 100]