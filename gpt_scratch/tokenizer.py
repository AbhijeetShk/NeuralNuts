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

print(stats)

ids = merge(ids, (1, 2), 256)

print(ids)