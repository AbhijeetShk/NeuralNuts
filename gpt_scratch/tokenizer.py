class BasicTokenizer:

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8")


tokenizer = BasicTokenizer()

text = "Hello, GPT! ❤️"

ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)
print(text == decoded)
# [72, 101, 108, 108, 111, 44, 32, 71, 80, 84, 33, 32, 226, 157, 164, 239, 184, 143]
# Hello, GPT! ❤️
# True


